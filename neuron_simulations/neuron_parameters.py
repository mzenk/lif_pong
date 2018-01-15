# some example neuron parameters
tutorial_params = {
    "cm"         : .2,
    "tau_m"      : 1.,
    "e_rev_E"    : 0.,
    "e_rev_I"    : -100.,
    "v_thresh"   : -50.,
    "tau_syn_E"  : 10.,
    "v_rest"     : -50.,
    "tau_syn_I"  : 10.,
    "v_reset"    : -50.001,
    "tau_refrac" : 10.,
    "i_offset"   : 0.,
}

tutorial_noise = {
    'rate_inh' : 3000.,
    'rate_exc' : 3000.,
    'w_exc'    : .001,
    'w_inh'    : -.001
}

# parameters from Mihai's thesis
sample_params = {
        "cm"         : .1,
        "tau_m"      : 20.,
        "e_rev_E"    : 0.,
        "e_rev_I"    : -90.,
        "v_thresh"   : -52.,
        "tau_syn_E"  : 10.,
        "v_rest"     : -65.,
        "tau_syn_I"  : 10.,
        "v_reset"    : -53.,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
}

noise_params = {
    'rate_inh' : 5000.,
    'rate_exc' : 5000.,
    'w_exc'    : .0035,
    'w_inh'    : -55./35 * .0035 # haven't understood yet how V_g is determined
}


# Dodo's params
dodo_params = {
        "cm"         : .1,
        "tau_m"      : 1.,
        "e_rev_E"    : 0.,
        "e_rev_I"    : -90.,
        "v_thresh"   : -52.,
        "tau_syn_E"  : 10.,
        "v_rest"     : -65.,
        "tau_syn_I"  : 10.,
        "v_reset"    : -53.,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
}

dodo_noise = {
    'rate_inh' : 2000.,
    'rate_exc' : 2000.,
    'w_exc'    : .001,
    'w_inh'    : -.0035
}

# Wei's params (from mixing paper; similar dynamics as Mihai's params)
wei_params = {
        "cm"         : .2,
        "tau_m"      : .1,
        "v_thresh"   : -50.,
        "v_rest"     : -65.,
        "v_reset"    : -50.01,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
        "e_rev_E"    : 0.,
        "e_rev_I"    : -100.,
        "tau_syn_E"  : 10.,
        "tau_syn_I"  : 10.,
}

wei_noise = {
    'rate_inh' : 400.,
    'rate_exc' : 400.,
    'w_exc'    : .001,
    'w_inh'    : -.001
}

wei_curr_params = {
        "cm"         : .2,
        "tau_m"      : .1,
        "tau_refrac" : 10.,
        "tau_syn_E"  : 10.,
        "tau_syn_I"  : 10.,
        "v_rest"     : -65.,
        "v_reset"    : -50.01,
        "v_thresh"   : -50.,
        "i_offset"   : 0.
}

wei_curr_noise = {
    'rate_inh' : 400.,
    'rate_exc' : 400.,
    'w_exc'    : .001,
    'w_inh'    : -.001
}
