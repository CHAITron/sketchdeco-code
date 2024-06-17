"""SAMPLING ONLY."""
import torch
import ptp_scripts.ptp_scripts as ptp
import ptp_scripts.ptp_utils as ptp_utils
from scripts.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
                steps,
                batch_size,
                shape,
                conditioning=None,
                inv_emb=None,
                x_T=None,
                unconditional_guidance_scale=1.,
                unconditional_conditioning=None,
                t_start=None,
                t_end=None,
                DPMencode=False,
                order=3,
                width=None,
                height=None,
                c2=False,
                top=None, 
                left=None, 
                bottom=None, 
                right=None,
                segmentation_map=None,
                target_height=None, 
                target_width=None,
                center_row_rm=None,
                center_col_rm=None,
                tau=0.4,
                **kwargs
                ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        if DPMencode:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )

            dpm_solver = DPM_Solver(model_fn, ns)
            data, _ = self.low_order_sample(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for step in range(order, steps + 1):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode)   

            return data['x'].to(device), None
        else:
            # x_T is a list
            model_fn_decode = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )
            model_fn_gen = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )
            
            controller1 = ptp.AttentionStore()
            controller2 = ptp.AttentionStore()
            self_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
            Inject_controller = ptp.AttentionStore()
            
            dpm_solver_decode = DPM_Solver(model_fn_decode, ns)
            dpm_solver_gen = DPM_Solver(model_fn_gen, ns)
            
            # decoded composited image
            ptp_utils.register_attention_control(self.model, controller1, center_row_rm, center_col_rm, target_height, target_width, 
                                                    width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
            c1, controller1 = self.low_order_sample(x[0], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=controller1)            
            # decoded composited image
            ptp_utils.register_attention_control(self.model, controller2, center_row_rm, center_col_rm, target_height, target_width, 
                                                width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
            c2, controller2 = self.low_order_sample(x[1], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=controller2)
            
            # decode for self-attention
            ptp_utils.register_attention_control(self.model, self_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), pseudo_cross=True)
            _, self_controller = self.low_order_sample(x[2], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode,
                                                                    controller=self_controller, ref_init=c2['x'].clone())
            
            # generation
            Inject_controller = [controller1, controller2, self_controller]
            ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), inject_bg=True)
            gen, _ = self.low_order_sample(x[3], dpm_solver_gen, steps, order, t_start, t_end, device, 
                                        DPMencode=DPMencode, controller=Inject_controller, inject=True)

            for i in range(len(c1['model_prev_list'])):
                blended = c1['model_prev_list'][i].clone() 
                gen['model_prev_list'][i] = blended.clone()
            
            del controller1, controller2, self_controller, gen_controller, Inject_controller
                        
            controller1 = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
                
            for step in range(order, steps + 1):
                # decoded composited image
                ptp_utils.register_attention_control(self.model, controller1, center_row_rm, center_col_rm, target_height, target_width, 
                                                    width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
                c1 = dpm_solver_decode.sample_one_step(c1, step, steps, order=order, DPMencode=DPMencode)
                controller=[controller1,None,None]
                
                if step < int(0.4*(steps) + 1 - order):
                    inject_bg = True
                else:
                    inject_bg = False
                    
                # generation
                ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, width, height, 
                                                    top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), inject_bg=inject_bg)
                gen = dpm_solver_gen.sample_one_step(gen, step, steps, order=order, DPMencode=DPMencode, controller=controller, inject=False)

                if step < int(tau*(steps) + 1 - order): 
                    blended = c1['x'].clone() 
                    gen['x'] = blended.clone()      
                    
            del controller1, gen_controller
            return gen['x'].to(device), None
            
    
    def low_order_sample(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False, controller=None, inject=False, ref_init=None):
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start
        
        total_controller = []
        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                    controller=[controller[0][0], controller[1][0], controller[2][0]] if isinstance(controller, list) else controller, 
                                                    inject=inject, ref_init=ref_init)]
            
            total_controller.append(controller)
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                            solver_type='dpmsolver', DPMencode=DPMencode)
                model_prev_list.append(dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                            controller=[controller[0][init_order], controller[1][init_order], controller[2][init_order]] if isinstance(controller, list) else controller,
                                                            inject=inject, ref_init=ref_init))
                total_controller.append(controller)
                t_prev_list.append(vec_t)
        
        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}, total_controller
    