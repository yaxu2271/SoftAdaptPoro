import torch
import torch.nn as nn
from pdb import set_trace as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def poro_loss (ip, ip_x, ip_xx, ip_y, ip_yy, iux, iux_x, iux_xx, iux_xy, iux_yy, iuy, iuy_xx, iuy_y, iuy_yx, iuy_yy, rp, rp_x, rp_xx, rp_y, rp_yy, rux, rux_x, rux_xx, rux_xy, rux_yy, ruy, ruy_xx, ruy_y, ruy_yx, ruy_yy, rF1x, iF1x, rF1y, iF1y, rF2, iF2, output1, output2, output3, output4, output5, output6):
    
    
    
    ## material parameter setup
    rho = 2.27
    rho_f = 1
    rho_a = 0.117
    w = 391
    
    mu_parameter = output1.to(device)
    lambda_parameter = output2.to(device)
    M = output3.to(device)
    phi = output4.to(device)
    rgamma = rho_a/(phi**2) + rho_f/phi
    kappa = output5.to(device)
    igamma = 1/(w*kappa)
    alpha = output6.to(device)


    ra = alpha - rho_f*rgamma/(rgamma**2+igamma**2)
    ia = rho_f*igamma/(rgamma**2+igamma**2)
    rb = rho - rho_f**2*rgamma/(rgamma**2+igamma**2)
    ib = rho_f**2*igamma/(rgamma**2+igamma**2)
    rc = rgamma/(rgamma**2+igamma**2)
    ic = -igamma/(rgamma**2+igamma**2)
    

    # original loss function
    e1 = ((mu_parameter*(rux_xx+rux_yy)+(lambda_parameter + mu_parameter)*(rux_xx+ruy_yx)-ra*rp_x+ia*ip_x+w**2*rb*rux-w**2*ib*iux)-rF1x)
    e2 = ((mu_parameter*(iux_xx+iux_yy)+(lambda_parameter + mu_parameter)*(iux_xx+iuy_yx)-ra*ip_x-ia*rp_x+w**2*rb*iux+w**2*ib*rux)-iF1x)
    e3 = ((mu_parameter*(ruy_xx+ruy_yy)+(lambda_parameter + mu_parameter)*(rux_xy+ruy_yy)-ra*rp_y+ia*ip_y+w**2*rb*ruy-w**2*ib*iuy)-rF1y)
    e4 = ((mu_parameter*(iuy_xx+iuy_yy)+(lambda_parameter + mu_parameter)*(iux_xy+iuy_yy)-ra*ip_y-ia*rp_y+w**2*rb*iuy+w**2*ib*ruy)-iF1y)
    e5 = ((1/w**2)*rc*(rp_xx+rp_yy)-(1/w**2)*ic*(ip_xx+ip_yy)+rp/M+ra*(rux_x+ruy_y)-ia*(iux_x+iuy_y)-rF2)
    e6 = ((1/w**2)*rc*(ip_xx+ip_yy)+(1/w**2)*ic*(rp_xx+rp_yy)+ip/M+ra*(iux_x+iuy_y)+ia*(rux_x+ruy_y)-iF2)

    
    e1_loss = torch.mean(e1**2)
    e2_loss = torch.mean(e2**2)
    e3_loss = torch.mean(e3**2)
    e4_loss = torch.mean(e4**2)
    e5_loss = torch.mean(e5**2)
    e6_loss = torch.mean(e6**2)
        


    return e1_loss, e2_loss, e3_loss, e4_loss, e5_loss, e6_loss
