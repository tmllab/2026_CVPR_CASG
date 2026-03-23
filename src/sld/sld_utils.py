SLD_HYPER_PARAMS = {
    'weak':{
        'sld_warmup_steps': 15,
        'sld_guidance_scale': 200,
        'sld_threshold': 0.0,
        'sld_momentum_scale': 0.0,
        'sld_mom_beta': 0.0
    },
    'medium':{
        'sld_warmup_steps': 10,
        'sld_guidance_scale': 1000,
        'sld_threshold': 0.01,
        'sld_momentum_scale': 0.3,
        'sld_mom_beta': 0.4
    },
    'strong':{
        'sld_warmup_steps': 7,
        'sld_guidance_scale': 2000,
        'sld_threshold': 0.025,
        'sld_momentum_scale': 0.5,
        'sld_mom_beta': 0.7
    },
    'max':{
        'sld_warmup_steps': 0,
        'sld_guidance_scale': 5000,
        'sld_threshold': 1.0,
        'sld_momentum_scale': 0.5,
        'sld_mom_beta': 0.7
    }
}
