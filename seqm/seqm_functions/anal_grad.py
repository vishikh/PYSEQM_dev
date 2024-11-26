import torch
from torch import pow, unsqueeze 
from .constants import a0
from .constants import ev
from .constants import sto6g_coeff, sto6g_exponent
from .cal_par import *
from .diat_overlap import diatom_overlap_matrix
from .two_elec_two_center_int_local_frame import two_elec_two_center_int_local_frame as TETCILF

# TODO: non of these tensors will need gradients (unless I wnat to do second derivatives), so I
# should find out what to specify in order to save memory on computations since the graph doesn't have to be stored
def scf_grad(P, const, mask, maskd, molsize, idxi,idxj, ni,nj,xij,rij, gam, parnuc,
            Z, gss,gpp,gp2,hsp, beta, zetas,zetap):
    # Xij is the vector from j to i in Angstroms
    # xij is the *unit* vector from j to i
    Xij = xij*rij.unsqueeze(1)*a0
    """
    Calculate the gradient of the ground state SCF energy
    in the units of ev/Angstrom
    The code follows the derivation outlined in 
    Dewar, Michael JS, and Yukio Yamaguchi. "Analytical first derivatives of the energy in MNDO." Computers & Chemistry 2.1 (1978): 25-29.
    https://doi.org/10.1016/0097-8485(78)80005-9
    """
    torch.set_printoptions(precision=6)

    dtype = Xij.dtype
    device = Xij.device
    nmol = P.shape[0]
    npairs=Xij.shape[0]
    qn_int = const.qn_int

    # Define the gradient tensor
    grad = torch.zeros(nmol*molsize, 3, dtype=dtype, device=device)

    # Overlap grad
    overlap_KAB_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    # overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij)

    # Finite diff overlap derivative gives more accurate resutls
    zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
    overlap_der_finiteDiff(overlap_KAB_x, idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int)

    # Core-core repulsion derivatives
    # First, derivative of g_AB
    tore = const.tore # Charges
    alpha = parnuc[0] 
    ZAZB = tore[ni]*tore[nj]
    pair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)
    g = core_core_der(alpha,rij,Xij,ZAZB,ni,nj,idxi,idxj,gam,pair_grad)

    # Two-center repulsion integral derivatives
    # Core-valence integral derivatives e1b_x and e2a_x also calculated as byproducts
    w_x  = torch.zeros(rij.shape[0],3,10,10,dtype=dtype, device=device)
    e1b_x,e2a_x = w_der(const,Z,tore,ni,nj,w_x,rij,xij,Xij,idxi,idxj,gss,gpp,gp2,hsp,zetas,zetap)

    # Assembly
    # P is currently in the shape of (nmol,4*molsize, 4*molsize)
    # I will reshape it to P0(nmol*molsize*molsize, 4, 4)
    P0 = P.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol*molsize*molsize,4,4)


    # ZAZB*g*d/dx(sasa|sbsb)
    pair_grad.add_((ZAZB*g).unsqueeze(1)*w_x[:,:,0,0])

    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    # sumKAB = torch.zeros(npairs,3,4,4,dtype=dtype, device=device)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5*P0[mask]
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[...,i,j] += torch.sum(Pp.unsqueeze(1)*(w_x[...,ind[i],:][...,:,ind[j]]),dim=(2,3))

    pair_grad.add_((P0[mask,None,:,:]*overlap_KAB_x).sum(dim=(2,3)))

    # Diagonal part: Multiply by 0.5 for energy
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P0[maskd[idxi]][..., indices[0], indices[1]] * weight).unsqueeze(-1)  # Shape: (npairs, 10, 1)
    PB = (P0[maskd[idxj]][..., indices[0], indices[1]] * weight).unsqueeze(-2)  # Shape: (npairs, 1, 10)
    
    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)

    # Collect in sumA and sumB tensors
    # reususe overlap_KAB_x here instead of creating new arrays
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    pair_grad.add_(0.5*torch.sum(P0[maskd[idxj]].unsqueeze(1) * sumA, dim=(2, 3)))
    sumB = overlap_KAB_x
    sumB.zero_()
    sumB[..., indices[0], indices[1]] = sumb
    pair_grad.add_(0.5*torch.sum(P0[maskd[idxi]].unsqueeze(1) * sumB, dim=(2, 3)))

    # Core-elecron interaction
    pair_grad.add_((P0[maskd[idxj],None,:,:]*e2a_x).sum(dim=(2,3)) + (P0[maskd[idxi],None,:,:]*e1b_x).sum(dim=(2,3)))

    grad.index_add_(0,idxi,pair_grad)
    grad.index_add_(0,idxj,pair_grad,alpha=-1.0)


    # grad = grad.reshape(nmol,molsize,3)
    print(f'Analytical gradient is:\n{grad.view(nmol,molsize,3)}')

def overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij):
    a0_sq = a0*a0

    # (sA|sB) overlap
    C_times_C = torch.einsum('bi,bj->bij',sto6g_coeff[qn_int[ni]-1,0],sto6g_coeff[qn_int[nj]-1,0])

    alpha1 = sto6g_exponent[qn_int[ni]-1,0,:]*(zetas[idxi].unsqueeze(1)**2)
    alpha2 = sto6g_exponent[qn_int[nj]-1,0,:]*(zetas[idxj].unsqueeze(1)**2)

    # alpha_i*alpha_j/(alpha_i+alpha_j)
    alpha_product = alpha1.unsqueeze(2) * alpha2.unsqueeze(1)  # Shape: (batch_size, vector_size, vector_size)
    alpha_sum = alpha1[..., None] + alpha2[..., None, :]
    alphas_1 = alpha_product / alpha_sum  # Shape: (batch_size, vector_size, vector_size)

    # <sA|sB>ij
    # From MOPAC the arugment of the exponential is not allowed to exceed -35 (presumably because exp(-35) ~ double precision minimum)
    sij = ((2.0*torch.div(torch.sqrt(alpha_product),alpha_sum))**(3/2))*torch.exp(-1.0*(alphas_1*(rij[:,None,None]**2)).clamp_(max=35.0))

    # d/dx of <sA|sB>
    sAsB = 2.0*alphas_1*sij

    # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
    # There is no dividing beta_mu+beta_nu by 2. Found this out during debugging.
    # Possibly because here we're only going over unique pairs, but in the total energy
    # expression the overlap term appears on the upper and lower triangle of Hcore 
    # and hence needs to be multiplied by 2.
    overlap_KAB_x[:,:,0,0] = ((beta[idxi,0]+beta[idxj,0])*torch.sum(C_times_C*sAsB,dim=(1,2))).unsqueeze(1)*Xij[:,:]/a0_sq 

    '''
    #(px|s)
    C_times_C = torch.einsum('bi,bj->bij',sto6g_coeff[qn_int[ni]-1,1],sto6g_coeff[qn_int[nj]-1,0])

    alpha1 = sto6g_exponent[qn_int[ni]-1,1,:]*(zetas[idxi].unsqueeze(1)**2)
    alpha2 = sto6g_exponent[qn_int[nj]-1,0,:]*(zetas[idxj].unsqueeze(1)**2)

    # alpha_i*alpha_j/(alpha_i+alpha_j)
    alpha_product = alpha1.unsqueeze(2) * alpha2.unsqueeze(1)  # Shape: (batch_size, vector_size, vector_size)
    alpha_sum = alpha1[..., None] + alpha2[..., None, :]
    alphas_1 = alpha_product / alpha_sum  # Shape: (batch_size, vector_size, vector_size)

    # <sA|sB>ij
    # From MOPAC the arugment of the exponential is not allowed to exceed -35 (presumably because exp(-35) ~ double precision minimum)
    sij = ((2.0*torch.div(torch.sqrt(alpha_product),alpha_sum))**(3/2))*torch.exp(-1.0*(alphas_1*(rij[:,None,None]**2)).clamp_(max=35.0))

    # d/dx of <sA|sB>
    sAsB = 2.0*alphas_1*sij

    # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
    # There is no dividing beta_mu+beta_nu by 2. Found this out during debugging.
    # Possibly because here we're only going over unique pairs, but in the total energy
    # expression the overlap term appears on the upper and lower triangle of Hcore 
    # and hence needs to be multiplied by 2.
    overlap_KAB_x[:,:,0,0] = ((beta[idxi,0]+beta[idxj,0])*torch.sum(C_times_C*sAsB,dim=(1,2))).unsqueeze(1)*Xij[:,:]/a0_sq 
    '''
    print(f'overlap_x from gaussians is \n{overlap_KAB_x}')


def core_core_der(alpha,rij,Xij,ZAZB,ni,nj,idxi,idxj,gam,pair_grad):
    rija=rij*a0
    # special case for N-H and O-H
    XH = ((ni==7) | (ni==8)) & (nj==1)
    t2 = torch.zeros_like(rij)
    tmp = torch.exp(-alpha[idxi]*rija)
    t2[~XH] = tmp[~XH]
    t2[XH] = tmp[XH]*rija[XH]
    t3 = torch.exp(-alpha[idxj]*rija)
    g = 1.0+t2+t3

    prefactor = alpha[idxi]
    prefactor[XH] = prefactor[XH]*rija[XH]-1.0
    t3 = alpha[idxj]*torch.exp(-alpha[idxj]*rija)
    coreTerm = ZAZB*gam/rija*(prefactor*tmp+t3)
    pair_grad[:,:] = coreTerm.unsqueeze(1)*Xij 
    return g

def w_der(const,Z,tore,ni,nj,w_x,rij,xij,Xij,idxi,idxj,gss,gpp,gp2,hsp,zetas,zetap):
    # Two-center repulsion integral derivatives
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)
    qn = const.qn
    hpp = 0.5*(gpp-gp2)
    qn0=qn[Z]
    isH = Z==1  # Hydrogen
    isX = Z>2   # Heavy atom
    rho_0=torch.zeros_like(qn0)
    rho_1=torch.zeros_like(qn0)
    rho_2=torch.zeros_like(qn0)
    dd=torch.zeros_like(qn0)
    qq=torch.zeros_like(qn0)
    rho1 = additive_term_rho1.apply
    rho2 = additive_term_rho2.apply

    dd[isX], qq[isX] = dd_qq(qn0[isX],zetas[isX], zetap[isX])
    rho_0[isH] = 0.5*ev/gss[isH]
    rho_0[isX] = 0.5*ev/gss[isX]
    if torch.sum(isX)>0:
        rho_1[isX] = rho1(hsp[isX],dd[isX])
        rho_2[isX] = rho2(hpp[isX],qq[isX])

    rho0a = rho_0[idxi]
    rho0b = rho_0[idxj]

    riHH_x, riXH_x, ri_x = der_TETCILF(w_x,const,ni, nj,xij, Xij, rij, dd[idxi], dd[idxj], qq[idxi], qq[idxj], rho_0[idxi], rho_0[idxj], rho_1[idxi], rho_1[idxj], rho_2[idxi], rho_2[idxj],tore)
    # d/dx (ss|ss)
    # (ss|ss) = riHH = ev/sqrt(r0[HH]**2+(rho0a[HH]+rho0b[HH])**2)
    # Why is rij in bohr? It should be in angstrom right? Ans: OpenMopac website seems to suggest using bohr as well
    # for the 2-e integrals.
    # Dividing by a0^2 for gradient in eV/ang
    # again assuming xij = xj-xi, and hence forgoing the minus sign

    ev_a02 = ev/a0/a0
    term_ss = ev_a02*pow(rij[HH]**2+(rho0a[HH]+rho0b[HH])**2,-1.5)
    w_x[HH,:,0,0] = term_ss.unsqueeze(1)*Xij[HH,:]

    # TODO: Derivatives of the rotation matrix
    if torch.any(isX):
        raise Exception("Analytical gradients not yet implemented for molecules with non-Hydrogen atoms")

    # Core-elecron interaction
    e1b_x = torch.zeros((rij.shape[0],3,4,4),dtype=w_x.dtype, device=w_x.device)
    e2a_x = torch.zeros((rij.shape[0],3,4,4),dtype=w_x.dtype, device=w_x.device)
    e1b_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]
    e2a_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]
    return e1b_x,e2a_x

from .constants import overlap_cutoff

def overlap_der_finiteDiff(overlap_KAB_x,idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int):
    overlap_pairs = rij <= overlap_cutoff
    delta = 5e-5  # TODO: Make sure this is a good delta (small enough, but still doesnt cause numerical instabilities)
    di_plus = torch.zeros(Xij.shape[0], 4, 4, dtype=Xij.dtype, device=Xij.device)
    di_minus = torch.clone(di_plus)
    # di_x = torch.zeros(Xij.shape[0], 3, 4, 4, dtype=Xij.dtype, device=Xij.device)
    for coord in range(3):
        # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
        Xij[:, coord] -= delta
        rij_ = torch.norm(Xij, dim=1)/a0
        xij_ = Xij / rij_.unsqueeze(1)
        di_plus[overlap_pairs] = diatom_overlap_matrix(
            ni[overlap_pairs],
            nj[overlap_pairs],
            xij_[overlap_pairs],
            rij_[overlap_pairs],
            zeta[idxi][overlap_pairs],
            zeta[idxj][overlap_pairs],
            qn_int,
        )
        Xij[:, coord] += 2.0 * delta
        rij_ = torch.norm(Xij, dim=1)/a0
        xij_ = Xij / rij_.unsqueeze(1)

        di_minus[overlap_pairs] = diatom_overlap_matrix(
            ni[overlap_pairs],
            nj[overlap_pairs],
            xij_[overlap_pairs],
            rij_[overlap_pairs],
            zeta[idxi][overlap_pairs],
            zeta[idxj][overlap_pairs],
            qn_int,
        )
        Xij[:, coord] -= delta
        # print(f'di_plus is\n{di_plus}')
        # print(f'di_minus is\n{di_minus}')

        overlap_KAB_x[:, coord, :, :] = (di_plus - di_minus) / (2.0 * delta)

    overlap_KAB_x[...,0,0] *= (beta[idxi,0]+beta[idxj,0]).unsqueeze(1)
    overlap_KAB_x[...,0,1:]  *= (beta[idxi,0:1]+beta[idxj,1:2]).unsqueeze(1)
    overlap_KAB_x[...,1:,0]  *= (beta[idxi,1:2]+beta[idxj,0:1]).unsqueeze(1)
    overlap_KAB_x[...,1:,1:] *= (beta[idxi,1:2,None]+beta[idxj,1:2,None]).unsqueeze(1)


def der_TETCILF(w_x_final,const,ni, nj,xij, Xij, r0, da0, db0, qa0, qb0, rho0a, rho0b, rho1a, rho1b, rho2a, rho2b,tore):

    print('WARNING: Do not recalculate 2e2c integrals. Save and resue from the scf run')
    riHH, riXH, ri, _, _, _ = \
           TETCILF(ni,nj,r0, tore, \
                da0, db0, qa0,qb0, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b)
    dtype = r0.dtype
    device = r0.device

    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)

    # Hydrogen - Hydrogen
    # aeeHH = (rho0a[HH]+rho0b[HH])**2
    term = -ev/a0/a0/r0.unsqueeze(1)*Xij
    ee = -r0*pow((r0**2+(rho0a+rho0b)**2),-1.5)
    ee_x = term*ee.unsqueeze(1)
    riHH_x = ee_x[HH,:]

    # Heavy atom - Hydrogen
    # aeeXH = (rho0a[XH]+rho0b[XH])**2
    rXH = r0[XH]
    daXH = da0[XH]
    qaXH = qa0[XH]*2.0
    adeXH = (rho1a[XH]+rho0b[XH])**2
    aqeXH = (rho2a[XH]+rho0b[XH])**2
    dsqr6XH = 2.0*rXH*pow(rXH**2 + aqeXH,-1.5)
    riXH_x = torch.zeros(XH.sum(),3,4,dtype=dtype, device=device)
    eeXH = ee[XH]
    riXH_x[...,1-1] = ee_x[XH,:]
    riXH_x[...,2-1] = -0.5*term[XH]*((rXH+daXH)*pow((rXH+daXH)**2+adeXH,-1.5) \
                   - (rXH-daXH)*pow((rXH-daXH)**2+adeXH,-1.5)).unsqueeze(1)
    riXH_x[...,3-1] = term[XH]*(eeXH + 0.25*(-(rXH+qaXH)*pow((rXH+qaXH)**2+aqeXH,-1.5) \
                       - (rXH-qaXH)*pow((rXH-qaXH)**2+aqeXH,-1.5) \
                       + dsqr6XH)).unsqueeze(1)
    riXH_x[...,4-1] = term[XH]*(eeXH + 0.25*(-2.0*rXH*pow(rXH**2+qaXH**2+aqeXH,-1.5) + dsqr6XH)).unsqueeze(1)

    # Heavy atom - Heavy atom
    term = term[XX]
    r =r0[XX]
    da = da0[XX]
    db = db0[XX]
    qa = qa0[XX]*2.0
    qb = qb0[XX]*2.0
    qa1 = qa0[XX]
    qb1 = qb0[XX]
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x =  torch.zeros(XX.sum(),3,22,dtype=dtype, device=device)

    # only the repeated terms are listed here
    ade = (rho1a[XX]+rho0b[XX])**2
    aqe = (rho2a[XX]+rho0b[XX])**2
    aed = (rho0a[XX]+rho1b[XX])**2
    aeq = (rho0a[XX]+rho2b[XX])**2
    axx = (rho1a[XX]+rho1b[XX])**2
    adq = (rho1a[XX]+rho2b[XX])**2
    aqd = (rho2a[XX]+rho1b[XX])**2
    aqq = (rho2a[XX]+rho2b[XX])**2
    ee  = ee[XX]
    dze = ((r+da)*pow((r+da)**2+ade,-1.5) \
                   - (r-da)*pow((r-da)**2+ade,-1.5))
    dsqr6 = 2.0*r*pow(r**2 + aqe,-1.5)
    qzze = -(r-qa)*pow((r-qa)**2 + aqe,-1.5) - (r+qa)*pow((r+qa)**2 + aqe,-1.5) + dsqr6
    qxxe = -2.0*r*pow(r**2 + qa**2 + aqe,-1.5) + dsqr6
    edz = (r-db)*pow((r-db)**2 + aed,-1.5) - (r+db)*pow((r+db)**2 + aed,-1.5)
    dsqr12 = 2.0*r*pow(r**2 + aeq,-1.5)
    eqzz = -(r-qb)*pow((r-qb)**2 + aeq,-1.5) - (r+qb)*pow((r+qb)**2 + aeq,-1.5) + dsqr12
    eqxx = -2.0*r*pow(r**2 + qb**2 + aeq,-1.5) + dsqr12
    dsqr20 = 2.0*(r+da)*pow((r+da)**2 + adq,-1.5)
    dsqr22 = 2.0*(r-da)*pow((r-da)**2 + adq,-1.5)
    dsqr24 = 2.0*(r-db)*pow((r-db)**2 + aqd,-1.5)
    dsqr26 = 2.0*(r+db)*pow((r+db)**2 + aqd,-1.5)
    dsqr36 = 4.0*(r)*pow(r**2 + aqq,-1.5)
    dsqr39 = 4.0*(r)*pow(r**2 + qa**2 + aqq,-1.5)
    dsqr40 = 4.0*(r)*pow(r**2 + qb**2 + aqq,-1.5)
    dsqr42 = 2.0*(r-qb)*pow((r-qb)**2 + aqq,-1.5)
    dsqr44 = 2.0*(r+qb)*pow((r+qb)**2 + aqq,-1.5)
    dsqr46 = 2.0*(r+qa)*pow((r+qa)**2 + aqq,-1.5)
    dsqr48 = 2.0*(r-qa)*pow((r-qa)**2 + aqq,-1.5)
    # all the index for ri is shfited by 1 to save space
    # C     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
    # C     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
    # C     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
    # C     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
    # C     (PP/P*P*)=21,   (P*P/P*P)=22
    ri_x[...,1-1] = ee_x[XX,:]
    ri_x[...,2-1] = -0.5*term*dze.unsqueeze(1)
    ri_x[...,3-1] = term*(ee + 0.25*qzze).unsqueeze(1)
    ri_x[...,4-1] = term*(ee + 0.25*qxxe).unsqueeze(1)
    ri_x[...,5-1] = -0.5*term*edz.unsqueeze(1)
    # RI(6) = DZDZ = EV2/SQR(16) + EV2/SQR(17) - EV2/SQR(18) - EV2/SQR(19)
    ri_x[...,6-1] = 0.25*term*(-(r+da-db)*pow((r+da-db)**2 + axx,-1.5) - (r-da+db)*pow((r-da+db)**2 + axx,-1.5) \
                  + (r-da-db)*pow((r-da-db)**2 + axx,-1.5) + (r+da+db)*pow((r+da+db)**2 + axx,-1.5)).unsqueeze(1)
    # RI(7) = DXDX = EV1/SQR(14) - EV1/SQR(15)
    ri_x[...,7-1] = 0.5*term*(-r*pow(r**2 + (da-db)**2 + axx,-1.5) +r*pow(r**2 + (da+db)**2 + axx,-1.5)).unsqueeze(1)
    # RI(8) = -EDZ -QZZDZ
    # QZZDZ = -EV3/SQR(32) + EV3/SQR(33) - EV3/SQR(34) + EV3/SQR(35)
    # + EV2/SQR(24) - EV2/SQR(26)
    ri_x[...,8-1] = -term*(0.5*edz + 0.125*((r+qa-db)*pow((r+qa-db)**2 + aqd,-1.5) - (r+qa+db)*pow((r+qa+db)**2 + aqd,-1.5) \
                + (r-qa-db)*pow((r-qa-db)**2 + aqd,-1.5) - (r-qa+db)*pow((r-qa+db)**2 + aqd,-1.5) \
                - dsqr24 + dsqr26)).unsqueeze(1)
    # RI(9) = -EDZ -QXXDZ
    # QXXDZ =  EV2/SQR(24) - EV2/SQR(25) - EV2/SQR(26) + EV2/SQR(27)
    ri_x[...,9-1] = -term*(0.5*edz + 0.125*(-dsqr24 + 2.0*(r-db)*pow((r-db)**2 + qa**2 + aqd,-1.5) \
                + dsqr26 - 2.0*(r+db)*pow((r+db)**2 + qa**2 + aqd,-1.5))).unsqueeze(1)

    # sqr(54)-sqr(72) use qa1 and qb1
    # RI(10) = -QXZDX
    # QXZDX = -EV2/SQR(58) + EV2/SQR(59) + EV2/SQR(60) - EV2/SQR(61)
    ri_x[...,10-1] = -0.25*term*((r+qa1)*pow((qa1-db)**2 + (r+qa1)**2 + aqd,-1.5) \
                   - (r-qa1)*pow((qa1-db)**2 + (r-qa1)**2 + aqd,-1.5) \
                   - (r+qa1)*pow((qa1+db)**2 + (r+qa1)**2 + aqd,-1.5) \
                   + (r-qa1)*pow((qa1+db)**2 + (r-qa1)**2 + aqd,-1.5)).unsqueeze(1)
    # RI(11) =  EE + EQZZ
    ri_x[...,11-1] = term*(ee + 0.25*eqzz).unsqueeze(1)
    # RI(12) =  EE + EQXX
    ri_x[...,12-1] = term*(ee + 0.25*eqxx).unsqueeze(1)
    # RI(13) = -DZE -DZQZZ
    # DZQZZ = -EV3/SQR(28) + EV3/SQR(29) - EV3/SQR(30) + EV3/SQR(31)
    #  - EV2/SQR(22) + EV2/SQR(20)
    ri_x[...,13-1] = -term*(0.5*dze + 0.125*(\
                 + (r+da-qb)*pow((r+da-qb)**2 + adq,-1.5) \
                 - (r-da-qb)*pow((r-da-qb)**2 + adq,-1.5) \
                 + (r+da+qb)*pow((r+da+qb)**2 + adq,-1.5) \
                 - (r-da+qb)*pow((r-da+qb)**2 + adq,-1.5) \
                 + dsqr22 - dsqr20)).unsqueeze(1)
    #
    # RI(14) = -DZE -DZQXX
    # DZQXX =  EV2/SQR(20) - EV2/SQR(21) - EV2/SQR(22) + EV2/SQR(23)
    ri_x[...,14-1] = -term*(0.5*dze + 0.125*(- dsqr20 + dsqr22 \
                 + 2.0*(r+da)*pow((r+da)**2 + qb**2 + adq,-1.5) \
                 - 2.0*(r-da)*pow((r-da)**2 + qb**2 + adq,-1.5))).unsqueeze(1)
    # RI(15) = -DXQXZ
    # DXQXZ = -EV2/SQR(54) + EV2/SQR(55) + EV2/SQR(56) - EV2/SQR(57)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[...,15-1] = -0.25*term*((r-qb1)*pow((da-qb1)**2 + (r-qb1)**2 + adq,-1.5) \
                   - (r+qb1)*pow((da-qb1)**2 + (r+qb1)**2 + adq,-1.5) \
                   - (r-qb1)*pow((da+qb1)**2 + (r-qb1)**2 + adq,-1.5) \
                   + (r+qb1)*pow((da+qb1)**2 + (r+qb1)**2 + adq,-1.5)).unsqueeze(1)
    # RI(16) = EE +EQZZ +QZZE +QZZQZZ
    # QZZQZZ = EV4/SQR(50) + EV4/SQR(51) + EV4/SQR(52) + EV4/SQR(53)
    # - EV3/SQR(48) - EV3/SQR(46) - EV3/SQR(42) - EV3/SQR(44)
    # + EV2/SQR(36)
    ri_x[...,16-1] = term*(ee + 0.25*eqzz + 0.25*qzze \
                 + 0.0625*(-(r+qa-qb)*pow((r+qa-qb)**2 + aqq,-1.5) \
                 - (r+qa+qb)*pow((r+qa+qb)**2 + aqq,-1.5) \
                 - (r-qa-qb)*pow((r-qa-qb)**2 + aqq,-1.5) \
                 - (r-qa+qb)*pow((r-qa+qb)**2 + aqq,-1.5) \
                 + dsqr48 + dsqr46 +dsqr42 + dsqr44 - dsqr36)).unsqueeze(1)
    # RI(17) = EE +EQZZ +QXXE +QXXQZZ
    # QXXQZZ = EV3/SQR(43) + EV3/SQR(45) - EV3/SQR(42) - EV3/SQR(44)
    #  - EV2/SQR(39) + EV2/SQR(36)
    ri_x[...,17-1] = term*(ee + 0.25*eqzz + 0.25*qxxe \
                 +0.0625*( -2.0*(r-qb)*pow((r-qb)**2 + qa**2 + aqq,-1.5) \
                 -2.0*(r+qb)*pow((r+qb)**2 + qa**2 + aqq,-1.5) \
                 + dsqr42 + dsqr44 + dsqr39 - dsqr36)).unsqueeze(1)
    # RI(18) = EE +EQXX +QZZE +QZZQXX
    # QZZQXX = EV3/SQR(47) + EV3/SQR(49) - EV3/SQR(46) - EV3/SQR(48)
    #  - EV2/SQR(40) + EV2/SQR(36)
    ri_x[...,18-1] = term*(ee + 0.25*eqxx + 0.25*qzze \
                 + 0.0625*(-2.0*(r+qa)*pow((r+qa)**2 + qb**2 + aqq,-1.5) \
                 -2.0*(r-qa)*pow((r-qa)**2 + qb**2 + aqq,-1.5) \
                 + dsqr46 + dsqr48 + dsqr40 - dsqr36)).unsqueeze(1)
    # RI(19) = EE +EQXX +QXXE +QXXQXX
    # QXXQXX = EV3/SQR(37) + EV3/SQR(38) - EV2/SQR(39) - EV2/SQR(40)
    # + EV2/SQR(36)
    qxxqxx = -2.0*r*pow(r**2 + (qa-qb)**2 + aqq,-1.5) \
            -2.0*r*pow(r**2 + (qa+qb)**2 + aqq,-1.5) \
           + dsqr39 + dsqr40 - dsqr36
    ri_x[...,19-1] = term*(ee + 0.25*eqxx + 0.25*qxxe + 0.0625*qxxqxx).unsqueeze(1)
    # RI(20) = QXZQXZ
    # QXZQXZ = EV3/SQR(65) - EV3/SQR(67) - EV3/SQR(69) + EV3/SQR(71)
    # - EV3/SQR(66) + EV3/SQR(68) + EV3/SQR(70) - EV3/SQR(72)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[...,20-1] = 0.125*term*(-(r+qa1-qb1)*pow((r+qa1-qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r+qa1+qb1)*pow((r+qa1+qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r-qa1-qb1)*pow((r-qa1-qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   - (r-qa1+qb1)*pow((r-qa1+qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r+qa1-qb1)*pow((r+qa1-qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   - (r+qa1+qb1)*pow((r+qa1+qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   - (r-qa1-qb1)*pow((r-qa1-qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   + (r-qa1+qb1)*pow((r-qa1+qb1)**2 + (qa1+qb1)**2 + aqq,-1.5)).unsqueeze(1)
    # RI(21) = EE +EQXX +QXXE +QXXQYY
    # QXXQYY = EV2/SQR(41) - EV2/SQR(39) - EV2/SQR(40) + EV2/SQR(36)
    qxxqyy = -4.0*r*pow(r**2 + qa**2 + qb**2 + aqq,-1.5) \
           + dsqr39 + dsqr40 - dsqr36
    ri_x[...,21-1] = term*(ee + 0.25*eqxx + 0.25*qxxe + 0.0625*qxxqyy).unsqueeze(1)
    # RI(22) = PP * (QXXQXX -QXXQYY)
    ri_x[...,22-1] = 0.03125* term * (qxxqxx - qxxqyy).unsqueeze(1)

    # # verify with finite difference
    # delta = 5e-5  # TODO: Make sure this is a good delta (small enough, but still doesnt cause numerical instabilities)
    # tore = const.tore
    # # ri_x_fd =  torch.zeros(XX.sum(),3,22,dtype=dtype, device=device)
    # # ri_x_fd =  torch.zeros(XH.sum(),3,4,dtype=dtype, device=device)
    # ri_x_fd =  torch.zeros(HH.sum(),3,dtype=dtype, device=device)
    # # di_x = torch.zeros(Xij.shape[0], 3, 4, 4, dtype=Xij.dtype, device=Xij.device)
    # for coord in range(3):
    #     # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
    #     Xij[:, coord] -= delta
    #     rij_ = torch.norm(Xij, dim=1)/a0
    #     ri_plus, _, _, _, _, _ = \
    #        TETCILF(ni,nj,rij_, tore, \
    #             da0, db0, qa0,qb0, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b)
    #     Xij[:, coord] += 2.0 * delta
    #     rij_ = torch.norm(Xij, dim=1)/a0
    #
    #     ri_minus, _, _, _, _, _ = \
    #        TETCILF(ni,nj,rij_, tore, \
    #             da0, db0, qa0,qb0, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b)
    #     Xij[:, coord] -= delta
    #
    #     # ri_x_fd[:, coord, :] = (ri_plus - ri_minus) / (2.0 * delta)
    #     ri_x_fd[:, coord] = (ri_plus - ri_minus) / (2.0 * delta)
    #
    # if not torch.allclose(riHH_x, ri_x_fd):
    #     # Find the differences
    #     diffs = torch.abs(riHH_x-ri_x_fd)
    #     print("Differences are :",diffs)
    #     # print("Values in tensor1 at these indices:", ri_x[differences])
    #     # print("Values in tensor2 at these indices:", ri_x_fd[differences])
    # else:
    #     print("Tensors are the same.")

    # We have the derivatives of the 2-center-2-elec integrals in the local frame
    # In the local frame for p-orbitals we have p-sigma (along the axis) ,p-pi,p-pi' (perpendicular to the axis)
    # But in the molecular frame we have px,py,pz which are rotations of p-sigma, p-pi, p-pi'
    # The p orbitals rotate just like the coordinate frame, so the rotation matrix is easy to express
    # We now make the rotation matrix and its derivative for the p-orbitals
    rot = torch.zeros(r0.shape[0],3,3)
    rot_der = torch.zeros(r0.shape[0],3,3,3)

    rxy2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 1])
    ryz2 = torch.square(Xij[:, 1]) + torch.square(Xij[:, 2])
    rxz2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 2])
    axis_tolerance = 1e-8
    onerij = 1.0/a0/r0

    Xalign = ryz2<axis_tolerance
    Yalign = rxz2<axis_tolerance
    Zalign = rxy2<axis_tolerance
    Noalign = ~(Xalign | Yalign | Zalign)

    # Rotation matrix row 1 is x,y,z
    # Rotation matrix row 2 is
    # Rotation matrix row 3 is

    xij_ = -xij[Noalign,...]
    rot[Noalign,0,:] = xij_
    onerxy = 1.0/torch.sqrt(rxy2[Noalign])
    rxy_over_rab = (torch.sqrt(rxy2)/r0)[Noalign]/a0
    rab_over_rxy = a0*r0[Noalign]*onerxy
    rab_over_rxy_sq = torch.square(rab_over_rxy)

    # The (1,0) element of the rotation matrix is -Y/sqrt(X^2+Y^2)*sign(X). If X (=xi-xj) is zero then there is a discontinuity in the sign function
    # and hence the derivative will not exist. So I'm printing a warning that there might be numerical errors here
    # Similaryly the (1,1) element of the rotation matrix is abs(X/sqrt(X^2+Y^2)). Again, the derivative of abs(X) will not exist when X=0, and hence this 
    # will lead to errors.
    if(xij_[:,0].any() == 0):
        print("WARNING: The x component of the pair distance is zero. This could lead to instabilities in the derivative of the rotation matrix")

    # As a quick-fix, I will add a small number (eps) when calculating sign(X) to avoid the aforementioned instability
    signcorrect = torch.sign(xij_[:,0]+torch.finfo(dtype).eps)
    rot[Noalign,1,0] = -xij_[:,1]*rab_over_rxy*signcorrect
    rot[Noalign,1,1] = torch.abs(xij_[:,0]*rab_over_rxy)

    rot[Noalign,2,0] = -xij_[:,0]*xij_[:,2]*rab_over_rxy
    rot[Noalign,2,1] = -xij_[:,1]*xij_[:,2]*rab_over_rxy
    rot[Noalign,2,2] = rxy_over_rab

    # Derivative of the rotation matrix
    termX = xij_[:,0]*onerij[Noalign]
    termY = xij_[:,1]*onerij[Noalign]
    termZ = xij_[:,2]*onerij[Noalign]
    # term = Xij[Noalign,:]*onerij.unsqueeze(1)
    rot_der[Noalign,0,0,0] = onerij[Noalign]-xij_[:,0]*termX
    rot_der[Noalign,0,0,1] = -xij_[:,0]*termY
    rot_der[Noalign,0,0,2] = -xij_[:,0]*termZ

    rot_der[Noalign,1,0,0] = -xij_[:,1]*termX
    rot_der[Noalign,1,0,1] = onerij[Noalign]-xij_[:,1]*termY
    rot_der[Noalign,1,0,2] = -xij_[:,1]*termZ

    rot_der[Noalign,2,0,0] = -xij_[:,2]*termX
    rot_der[Noalign,2,0,1] = -xij_[:,2]*termY
    rot_der[Noalign,2,0,2] = onerij[Noalign]-xij_[:,2]*termZ

    rot_der[Noalign,0,2,2] = xij_[:,0]*onerxy - rot[Noalign,2,2]*termX
    rot_der[Noalign,1,2,2] = xij_[:,1]*onerxy - rot[Noalign,2,2]*termY
    rot_der[Noalign,2,2,2] = -rot[Noalign,2,2]*termZ

    rot_der[Noalign,0,1,0] = -rot[Noalign,1,1]*rot[Noalign,1,0]*onerxy
    rot_der[Noalign,1,1,0] = -torch.square(rot[Noalign,1,1])*onerxy
    # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # tolerance = 1e-8
    # assert torch.allclose(rot_der[Noalign,0,1,0],-rot_der[Noalign,1,0,0]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.allclose(rot_der[Noalign,1,1,0],-rot_der[Noalign,1,0,1]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.all(torch.abs(-rot_der[Noalign,1,0,2]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)

    rot_der[Noalign,0:2,1,0] *= signcorrect.unsqueeze(1)

    rot_der[Noalign,0,1,1] = torch.square(rot[Noalign,1,0])*onerxy
    rot_der[Noalign,1,1,1] = rot[Noalign,1,1]*rot[Noalign,1,0]*onerxy
    # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # tolerance = 1e-8
    # mopacs = rot_der[Noalign,0,0,0]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq
    # mine = rot_der[Noalign,0,1,1]
    # assert torch.allclose(mine,mopacs,atol=tolerance)
    # assert torch.allclose(rot_der[Noalign,1,1,1],rot_der[Noalign,0,0,1]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.all(torch.abs(rot_der[Noalign,0,0,2]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)

    rot_der[Noalign,0:2,1,1] *= signcorrect.unsqueeze(1)

    rot_der[Noalign,0,2,0] = -xij_[:,2]*rot_der[Noalign,0,0,0]*rab_over_rxy -xij_[:,0]*rot_der[Noalign,2,0,0]*rab_over_rxy + xij_[:,0]*xij_[:,2]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq
    rot_der[Noalign,1,2,0] = torch.prod(xij_,dim=1)*(onerxy+rab_over_rxy_sq*onerxy)
    rot_der[Noalign,2,2,0] = -termX*rxy_over_rab

    rot_der[Noalign,0,2,1] = rot_der[Noalign,1,2,0]
    rot_der[Noalign,1,2,1] = -xij_[:,2]*rot_der[Noalign,1,0,1]*rab_over_rxy -xij_[:,1]*rot_der[Noalign,2,0,1]*rab_over_rxy + xij_[:,1]*xij_[:,2]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq
    rot_der[Noalign,2,2,1] = -termY*rxy_over_rab

    # # verify with finite difference
    # delta = 5e-5  # TODO: Make sure this is a good delta (small enough, but still doesnt cause numerical instabilities)
    # rot_der_fd = torch.zeros(r0.shape[0],3,3,3)
    # for coord in range(3):
    #     # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
    #     Xij[:, coord] -= delta
    #     rij_ = torch.norm(Xij, dim=1)/a0
    #     xij = Xij/torch.norm(Xij,dim=1).unsqueeze(1)
    #     rot_plus = makeRotMat(rij_,Xij,xij)
    #     Xij[:, coord] += 2.0 * delta
    #     rij_ = torch.norm(Xij, dim=1)/a0
    #     xij = Xij/torch.norm(Xij,dim=1).unsqueeze(1)
    #     rot_minus = makeRotMat(rij_,Xij,xij)
    #
    #     Xij[:, coord] -= delta
    #
    #     rot_der_fd[:, coord,...] = (rot_plus - rot_minus) / (2.0 * delta)
    #
    # if not torch.allclose(rot_der, rot_der_fd,atol=1e-9):
    #     # Find the differences
    #     diffs = torch.abs(rot_der-rot_der_fd)
    #     differences = torch.where(diffs>1e-9)
    #     print("Differences are at indices:",differences)
    #     print("Differences are :",diffs[differences])
    #     # print("Values in tensor1 at these indices:", ri_x[differences])
    #     # print("Values in tensor2 at these indices:", ri_x_fd[differences])
    # else:
    #     print("Tensors are the same.")

    rot[Zalign, 0, 2] = torch.sign(-xij[Zalign,0]) 
    rot[Zalign, 1, 1] = 1.0
    rot[Zalign, 0, 0] = rot[Zalign, 0, 2]
    rot_der[Zalign, 0, 0, 0] = onerij[Zalign]
    rot_der[Zalign, 0, 2, 2] = -onerij[Zalign]
    rot_der[Zalign, 1, 0, 1] = onerij[Zalign]
    rot_der[Zalign, 1, 1, 2] = -rot[Zalign, 0, 2]*onerij[Zalign]

    rot[Xalign, 0, 0] = torch.sign(-xij[Xalign,0]) 
    rot[Xalign, 1, 1] = rot[Xalign, 0, 2]
    rot[Xalign, 2, 2] = 1.0
    rot_der[Xalign, 1, 0, 1] = onerij[Xalign]
    rot_der[Xalign, 1, 1, 0] = -onerij[Xalign]
    rot_der[Xalign, 2, 0, 2] = onerij[Xalign]
    rot_der[Xalign, 2, 2, 0] = -rot[Zalign, 0, 0]*onerij[Zalign]

    rot[Xalign, 0, 1] = torch.sign(-xij[Xalign,0]) 
    rot[Xalign, 1, 0] = -rot[Xalign, 0, 1]
    rot[Xalign, 2, 2] = 1.0
    rot_der[Xalign, 0, 0, 0] = onerij[Xalign]
    rot_der[Xalign, 0, 1, 1] = onerij[Xalign]
    rot_der[Xalign, 2, 0, 2] = onerij[Xalign]
    rot_der[Xalign, 2, 2, 1] = -rot[Zalign, 0, 1]*onerij[Zalign]

    w_x = torch.empty(ri.shape[0],3,100,device=device,dtype=dtype)
    idx=-1
    for k in range(0,4):
        for l in range(0,k+1):
             for m in range(0,4):
                 for n in range(0,m+1):
                     idx = idx + 1
                     if (k==0):
                         if m==0:
                             # (ss|ss)
                             w_x[...,0] = ri_x[...,0]
                         elif n==0:
                             # (ss|ps)
                             w_x[...,idx] = ri_x[...,4]*rot[XX,None,0,m] + ri[:,None,4]*rot_der[XX,:,0,m]
                         else:
                             # (ss|pp)
                             w_x[...,idx] = ri_x[...,10]*(rot[XX,0,m]*rot[XX,0,n]).unsqueeze(1) +\
                                     ri[:,None,10]*(rot_der[XX,:,0,m]*rot[XX,None,0,n]+rot[XX,None,0,m]*rot_der[XX,:,0,n]) + \
                                            ri_x[...,11]*(rot[XX,1,m]*rot[XX,1,n]+rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1) +\
                                            ri[:,None,11]*(rot_der[XX,:,1,m]*rot[XX,None,1,n]+rot_der[XX,:,2,m]*rot[XX,None,2,n]+
                                                           rot[XX,None,1,m]*rot_der[XX,:,1,n]+rot[XX,None,2,m]*rot_der[XX,:,2,n])

                     elif l==0:
                         if m==0:
                             # (ps|ss)
                             w_x[...,idx] = ri_x[...,1]*rot[XX,None,0,k] + ri[:,None,1]*rot_der[XX,:,0,k]
                         elif n==0:
                             # (ps|ps)
                             w_x[...,idx] = ri_x[...,5]*(rot[XX,0,k]*rot[XX,0,m]).unsqueeze(1) +\
                                     ri[:,None,5]*(rot_der[XX,:,0,k]*rot[XX,None,0,m]+rot[XX,None,0,k]*rot_der[XX,:,0,m]) + \
                                            ri_x[...,6]*(rot[XX,1,k]*rot[XX,1,m]+rot[XX,2,k]*rot[XX,2,m]).unsqueeze(1) +\
                                            ri[:,None,6]*(rot_der[XX,:,1,k]*rot[XX,None,1,m]+rot_der[XX,:,2,k]*rot[XX,None,2,m]+
                                                           rot[XX,None,1,k]*rot_der[XX,:,1,m]+rot[XX,None,2,k]*rot_der[XX,:,2,m])
                         else:
                             #(ps|pp)
                             w_x[...,idx] = ri_x[...,12]*(rot[XX,0,k]*rot[XX,0,n]*rot[XX,0,m]).unsqueeze(1) +\
                                     ri[:,None,12]*(rot_der[XX,:,0,k]*(rot[XX,0,n]*rot[XX,0,m]).unsqueeze(1) +
                                      rot_der[XX,:,0,n]*(rot[XX,0,k]*rot[XX,0,m]).unsqueeze(1)+rot_der[XX,:,0,m]*(rot[XX,0,n]*rot[XX,0,k]).unsqueeze(1)) +\
                                     ri_x[...,13]*((rot[XX,1,m]*rot[XX,1,n]+rot[XX,2,m]*rot[XX,2,n])*rot[XX,0,k]).unsqueeze(1) +\
                                     ri[:,None,13]*((rot[XX,1,m]*rot[XX,1,n]+rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)*rot_der[XX,:,0,k]+
                                                   (rot_der[XX,:,1,m]*rot[XX,None,1,n]+rot[XX,None,1,m]*rot_der[XX,:,1,n]+
                                                    rot_der[XX,:,2,m]*rot[XX,None,2,n]+rot[XX,None,2,m]*rot_der[XX,:,2,n])*rot[XX,None,0,k]) +\
                                     ri_x[...,14]*(rot[XX,1,k]*(rot[XX,1,n]*rot[XX,0,m]+rot[XX,1,m]*rot[XX,0,n])+
                                                  rot[XX,2,k]*(rot[XX,2,m]*rot[XX,0,n]+rot[XX,2,n]*rot[XX,0,m])).unsqueeze(1) +\
                                     ri[:,None,14]*(rot_der[XX,:,1,k]*(rot[XX,1,n]*rot[XX,0,m]+rot[XX,1,m]*rot[XX,0,n]).unsqueeze(1)+
                                                   rot[XX,None,1,k]*(rot_der[XX,:,1,m]*rot[XX,None,0,n]+rot[XX,None,1,m]*rot_der[XX,:,0,n]+
                                                                     rot_der[XX,:,1,n]*rot[XX,None,0,m]+rot[XX,None,1,n]*rot_der[XX,:,0,m])+
                                                   rot_der[XX,:,2,k]*(rot[XX,2,n]*rot[XX,0,m]+rot[XX,2,m]*rot[XX,0,n]).unsqueeze(1)+
                                                   rot[XX,None,2,k]*(rot_der[XX,:,2,n]*rot[XX,None,0,m]+rot[XX,None,2,n]*rot_der[XX,:,0,m]+
                                                                     rot_der[XX,:,2,m]*rot[XX,None,0,n]+rot[XX,None,2,m]*rot_der[XX,:,0,n]))
                             pass
                     else:
                         if m==0:
                             # (pp|ss)
                             w_x[...,idx] = ri_x[...,2]*(rot[XX,0,k]*rot[XX,0,l]).unsqueeze(1) +\
                                     ri[:,None,2]*(rot_der[XX,:,0,k]*rot[XX,None,0,l]+rot[XX,None,0,k]*rot_der[XX,:,0,l]) + \
                                            ri_x[...,3]*(rot[XX,1,k]*rot[XX,1,l]+rot[XX,2,k]*rot[XX,2,l]).unsqueeze(1) +\
                                            ri[:,None,3]*(rot_der[XX,:,1,k]*rot[XX,None,1,l]+rot_der[XX,:,2,k]*rot[XX,None,2,l]+
                                                           rot[XX,None,1,k]*rot_der[XX,:,1,l]+rot[XX,None,2,k]*rot_der[XX,:,2,l])
                         elif n==0:
                             # (pp|ps)
                             w_x[...,idx] = ri_x[...,7]*(rot[XX,0,k]*rot[XX,0,l]*rot[XX,0,m]).unsqueeze(1) +\
                                     ri[:,None,7]*(rot_der[XX,:,0,k]*(rot[XX,0,l]*rot[XX,0,m]).unsqueeze(1) +
                                      rot_der[XX,:,0,l]*(rot[XX,0,k]*rot[XX,0,m]).unsqueeze(1)+rot_der[XX,:,0,m]*(rot[XX,0,l]*rot[XX,0,k]).unsqueeze(1)) +\
                                     ri_x[...,8]*((rot[XX,1,k]*rot[XX,1,l]+rot[XX,2,k]*rot[XX,2,l])*rot[XX,0,m]).unsqueeze(1) +\
                                     ri[:,None,8]*((rot[XX,1,k]*rot[XX,1,l]+rot[XX,2,k]*rot[XX,2,l]).unsqueeze(1)*rot_der[XX,:,0,m]+
                                                   (rot_der[XX,:,1,k]*rot[XX,None,1,l]+rot[XX,None,1,k]*rot_der[XX,:,1,l]+
                                                    rot_der[XX,:,2,k]*rot[XX,None,2,l]+rot[XX,None,2,k]*rot_der[XX,:,2,l])*rot[XX,None,0,m]) +\
                                     ri_x[...,9]*(rot[XX,0,k]*(rot[XX,1,l]*rot[XX,1,m]+rot[XX,2,l]*rot[XX,2,m])+
                                                  rot[XX,0,l]*(rot[XX,1,k]*rot[XX,1,m]+rot[XX,2,k]*rot[XX,2,m])).unsqueeze(1) +\
                                     ri[:,None,9]*(rot_der[XX,:,0,k]*(rot[XX,1,l]*rot[XX,1,m]+rot[XX,2,l]*rot[XX,2,m]).unsqueeze(1)+
                                                   rot[XX,None,0,k]*(rot_der[XX,:,1,l]*rot[XX,None,1,m]+rot[XX,None,2,l]*rot_der[XX,:,2,m]+
                                                                     rot_der[XX,:,1,m]*rot[XX,None,1,l]+rot[XX,None,2,m]*rot_der[XX,:,2,l])+
                                                   rot_der[XX,:,0,l]*(rot[XX,1,k]*rot[XX,1,m]+rot[XX,2,k]*rot[XX,2,m]).unsqueeze(1)+
                                                   rot[XX,None,0,l]*(rot_der[XX,:,1,k]*rot[XX,None,1,m]+rot[XX,None,2,k]*rot_der[XX,:,2,m]+
                                                                     rot_der[XX,:,1,m]*rot[XX,None,1,k]+rot[XX,None,2,m]*rot_der[XX,:,2,k]))

                         else:
                             #(pp|pp)
                             w_x[...,idx] = ri_x[...,16-1] * (rot[XX,0,k] * rot[XX,0,l] * rot[XX,0,m] * rot[XX,0,n]).unsqueeze(1) +  \
              ri[:,None,16-1] * (rot_der[XX,:,0,k]*(rot[XX,0,l]*rot[XX,0,m]*rot[XX,0,n]).unsqueeze(1)+\
              rot_der[XX,:,0,l]*(rot[XX,0,k]*rot[XX,0,m]*rot[XX,0,n]).unsqueeze(1)+(rot_der[XX,:,0,m]*rot[XX,0,k]*rot[XX,0,l]*rot[XX,0,n]).unsqueeze(1)+\
              (rot[XX,0,k]*rot[XX,0,l]*rot[XX,0,m]).unsqueeze(1)*rot_der[XX,:,0,n]) + ri_x[...,17-1] * ((rot[XX,1,k]*rot[XX,1,l]+\
              rot[XX,2,k]*rot[XX,2,l]) * rot[XX,0,m] * rot[XX,0,n]).unsqueeze(1) + ri[:,None,17-1] * ((rot_der[XX,:,1,k]*rot[XX,None,1,l]+\
              rot[XX,None,1,k]*rot_der[XX,:,1,l]+rot_der[XX,:,2,k]*rot[XX,None,2,l]+rot[XX,None,2,k]*rot_der[XX,:,2,l])*(rot[XX,0,m]*rot[XX,0,n]).unsqueeze(1)+\
              (rot[XX,1,k]*rot[XX,1,l]+rot[XX,2,k]*rot[XX,2,l]).unsqueeze(1)*(rot_der[XX,:,0,m]*rot[XX,None,0,n]+rot[XX,None,0,m]*rot_der[XX,:,0,n])) +\
               ri_x[...,18-1] * (rot[XX,0,k] * rot[XX,0,l] * (rot[XX,1,m]*rot[XX,1,n]+rot[XX,2,m]*rot[XX,2,n])).unsqueeze(1) + \
              ri[:,None,18-1] * ((rot_der[XX,:,0,k]*rot[XX,None,0,l]+rot[XX,None,0,k]*rot_der[XX,:,0,l])*(rot[XX,1,m]*rot[XX,1,n]+\
              rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)+(rot[XX,0,k]*rot[XX,0,l]).unsqueeze(1)*(rot_der[XX,:,1,m]*rot[XX,None,1,n]+rot[XX,None,1,m]*rot_der[XX,:,1,n]+\
              rot_der[XX,:,2,m]*rot[XX,None,2,n]+rot[XX,None,2,m]*rot_der[XX,:,2,n]))
                             w_x[...,idx] += ri_x[...,19-1] * (rot[XX,1,k]*rot[XX,1,l]*rot[XX,1,m]*rot[XX,1,n]+\
              rot[XX,2,k]*rot[XX,2,l]*rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1) + ri[:,None,19-1] * \
              (rot_der[XX,:,1,k]*(rot[XX,1,l]*rot[XX,1,m]*rot[XX,1,n]).unsqueeze(1)+rot_der[XX,:,1,l]*(rot[XX,1,k]*rot[XX,1,m]*rot[XX,1,n]).unsqueeze(1)+\
              rot_der[XX,:,1,m]*(rot[XX,1,k]*rot[XX,1,l]*rot[XX,1,n]).unsqueeze(1)+(rot[XX,1,k]*rot[XX,1,l]*rot[XX,1,m]).unsqueeze(1)*rot_der[XX,:,1,n]+\
              rot_der[XX,:,2,k]*(rot[XX,2,l]*rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)+rot_der[XX,:,2,l]*(rot[XX,2,k]*rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)+\
              rot_der[XX,:,2,m]*(rot[XX,2,k]*rot[XX,2,l]*rot[XX,2,n]).unsqueeze(1)+(rot[XX,2,k]*rot[XX,2,l]*rot[XX,2,m]).unsqueeze(1)*rot_der[XX,:,2,n]) + \
              ri_x[...,20-1] * (rot[XX,0,k]*(rot[XX,0,m]*(rot[XX,1,l]*rot[XX,1,n]+rot[XX,2,l]*rot[XX,2,n])+\
              rot[XX,0,n]*(rot[XX,1,l]*rot[XX,1,m]+rot[XX,2,l]*rot[XX,2,m]))+\
              rot[XX,0,l]*(rot[XX,0,m]*(rot[XX,1,k]*rot[XX,1,n]+rot[XX,2,k]*rot[XX,2,n])+\
              rot[XX,0,n]*(rot[XX,1,k]*rot[XX,1,m]+rot[XX,2,k]*rot[XX,2,m]))).unsqueeze(1)
                 #      TO AVOID COMPILER DIFFICULTIES THIS IS DIVIDED
                             temp1 = rot_der[XX,:,0,k] * (rot[XX,0,m]*(rot[XX,1,l]*rot[XX,1,n]+rot[XX,2,l]*rot[XX,2,n])+\
              rot[XX,0,n]*(rot[XX,1,l]*rot[XX,1,m]+rot[XX,2,l]*rot[XX,2,m])) + rot_der[XX,:,0,l] * \
              (rot[XX,0,m]*(rot[XX,1,k]*rot[XX,1,n]+rot[XX,2,k]*rot[XX,2,n])+rot[XX,0,n]*(rot[XX,1,k]*rot[XX,1,m]+\
              rot[XX,2,k]*rot[XX,2,m])) + rot[XX,0,k] * (rot_der[XX,:,0,m]*(rot[XX,1,l]*rot[XX,1,n]+\
              rot[XX,2,l]*rot[XX,2,n])+rot_der[XX,:,0,n]*(rot[XX,1,l]*rot[XX,1,m]+rot[XX,2,l]*rot[XX,2,m])) + rot[XX,0,l] \
              * (rot_der[XX,:,0,m]*(rot[XX,1,k]*rot[XX,1,n]+rot[XX,2,k]*rot[XX,2,n])+rot_der[XX,:,0,n]*(rot[XX,1,k]*rot[XX,1,m]+\
              rot[XX,2,k]*rot[XX,2,m]))
                             temp2 = rot[XX,0,k] * (rot[XX,0,m]*(rot_der[XX,:,1,l]*rot[XX,1,n]+rot[XX,1,l]*rot_der[XX,:,1,n]+\
              rot_der[XX,:,2,l]*rot[XX,2,n]+rot[XX,2,l]*rot_der[XX,:,2,n])+rot[XX,0,n]*(rot_der[XX,:,1,l]*rot[XX,1,m]+\
              rot[XX,1,l]*rot_der[XX,:,1,m]+rot_der[XX,:,2,l]*rot[XX,2,m]+rot[XX,2,l]*rot_der[XX,:,2,m])) + rot[XX,0,l] * \
              (rot[XX,0,m]*(rot_der[XX,:,1,k]*rot[XX,1,n]+rot[XX,1,k]*rot_der[XX,:,1,n]+rot_der[XX,:,2,k]*rot[XX,2,n]+\
              rot[XX,2,k]*rot_der[XX,:,2,n])+rot[XX,0,n]*(rot_der[XX,:,1,k]*rot[XX,1,m]+rot[XX,1,k]*rot_der[XX,:,1,m]+\
              rot_der[XX,:,2,k]*rot[XX,2,m]+rot[XX,2,k]*rot_der[XX,:,2,m]))
                             w_x[...,idx] += ri[:,None,20-1] * (temp1+temp2).unsqueeze(1)
                             w_x[...,idx] += ri_x[...,21-1] * (rot[XX,1,k]*rot[XX,1,l]*rot[XX,2,m]*rot[XX,2,n]+\
              rot[XX,2,k]*rot[XX,2,l]*rot[XX,1,m]*rot[XX,1,n]).unsqueeze(1) + ri[:,None,21-1] * \
              (rot_der[XX,:,1,k]*(rot[XX,1,l]*rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)+rot_der[XX,:,1,l]*(rot[XX,1,k]*rot[XX,2,m]*rot[XX,2,n]).unsqueeze(1)+\
              rot_der[XX,:,2,m]*(rot[XX,1,k]*rot[XX,1,l]*rot[XX,2,n]).unsqueeze(1)+(rot[XX,1,k]*rot[XX,1,l]*rot[XX,2,m]).unsqueeze(1)*rot_der[XX,:,2,n]+\
              rot_der[XX,:,2,k]*(rot[XX,2,l]*rot[XX,1,m]*rot[XX,1,n]).unsqueeze(1)+rot_der[XX,:,2,l]*(rot[XX,2,k]*rot[XX,1,m]*rot[XX,1,n]).unsqueeze(1)+\
              rot_der[XX,:,1,m]*(rot[XX,2,k]*rot[XX,2,l]*rot[XX,1,n]).unsqueeze(1)+(rot[XX,2,k]*rot[XX,2,l]*rot[XX,1,m]).unsqueeze(1)*rot_der[XX,:,1,n])
                             w_x[...,idx] += ri_x[...,22-1] * ((rot[XX,1,k]*rot[XX,2,l]+rot[XX,2,k]*rot[XX,1,l]) * \
              (rot[XX,1,m]*rot[XX,2,n]+rot[XX,2,m]*rot[XX,1,n])).unsqueeze(1) + ri[:,None,22-1] * ((rot_der[XX,:,1,k]*rot[XX,None,2,l]+\
              rot[XX,None,1,k]*rot_der[XX,:,2,l]+rot_der[XX,:,2,k]*rot[XX,None,1,l]+rot[XX,None,2,k]*rot_der[XX,:,1,l])*(rot[XX,1,m]*rot[XX,2,n]+\
              rot[XX,2,m]*rot[XX,1,n]).unsqueeze(1)+ (rot[XX,1,k]*rot[XX,2,l]+rot[XX,2,k]*rot[XX,1,l]).unsqueeze(1)*(rot_der[XX,:,1,m]*rot[XX,None,2,n]+\
              rot[XX,None,1,m]*rot_der[XX,:,2,n]+rot_der[XX,:,2,m]*rot[XX,None,1,n]+rot[XX,None,2,m]*rot_der[XX,:,1,n]))
                     

    return riHH_x, riXH_x, ri_x

def makeRotMat(r0,Xij,xij):
    rot = torch.zeros(r0.shape[0],3,3)

    rxy2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 1])
    ryz2 = torch.square(Xij[:, 1]) + torch.square(Xij[:, 2])
    rxz2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 2])
    axis_tolerance = 1e-8

    Xalign = ryz2<axis_tolerance
    Yalign = rxz2<axis_tolerance
    Zalign = rxy2<axis_tolerance
    Noalign = ~(Xalign | Yalign | Zalign)

    # Rotation matrix row 1 is x,y,z
    # Rotation matrix row 2 is
    # Rotation matrix row 3 is

    xij_ = -xij[Noalign,...]
    rot[Noalign,0,:] = xij_
    onerxy = 1.0/torch.sqrt(rxy2[Noalign])
    rxy_over_rab = (torch.sqrt(rxy2)/r0)[Noalign]/a0
    rab_over_rxy = a0*r0[Noalign]*onerxy

    # When rot[Noalign,0,0] is zero, torch.sign gives a zero. Since I multiply this sign to other elements they become zero as well.
    # To avoid this, I'm adding a small value (eps)
    signcorrect = torch.sign(xij_[:,0]+torch.finfo(xij.dtype).eps)
    rot[Noalign,1,0] = -xij_[:,1]*rab_over_rxy*signcorrect
    rot[Noalign,1,1] = torch.abs(xij_[:,0]*rab_over_rxy)

    rot[Noalign,2,0] = -xij_[:,0]*xij_[:,2]*rab_over_rxy
    rot[Noalign,2,1] = -xij_[:,1]*xij_[:,2]*rab_over_rxy
    rot[Noalign,2,2] = rxy_over_rab

    return rot

