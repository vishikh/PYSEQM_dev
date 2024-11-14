import torch
from .constants import a0
from .constants import ev
from .constants import sto6g_coeff, sto6g_exponent

def scf_grad(P, const, mask, maskd, molsize, idxi,idxj, ni,nj,xij, Xij,rij, gam, parnuc,
            Z, gss, beta, zetas,zetap):
    # Xij is the vector from j to i in Angstroms
    # xij is the *unit* vector from j to i
    """
    Calculate the gradient of the ground state SCF energy
    in the units of ev/Angstrom
    """
    print("Reached gradient")
    # print(f'Vishikh: shape of P is {P.shape}')

    dtype = Xij.dtype
    device = Xij.device
    nmol = P.shape[0]
    npairs=Xij.shape[0]
    qn_int = const.qn_int
    a0_sq = a0*a0
    print(f'There are {nmol} molecules with {molsize} atoms in each molecule')
    # print(f'The convered density is \n{P} whose shape is {P.shape}')

    # Define the gradient tensor
    grad = torch.zeros(nmol*molsize, 3, dtype=dtype, device=device)

    # Overlap grad
    # TODO: Add a cutoff distance for overlap
    overlap_ab_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    # overlap_ab_x([pair_number], [x_der,y_der,z_der],[sA,pxA,pyA,pzA],[sB,pxB,pyB,pzB])

    # print(f'zeta_s is\n{zetas}')
    # print(f'Principal quantum no of ni is\n{qn_int[ni]}')
    # print(f'Principal quantum no of nj is\n{qn_int[nj]}')
    max_l = 1 #only doing s
    for i in range(npairs):
        for l in range(max_l):
            # print(f'{sto6g_coeff[qn_int[ni[i]]-1,l,:]}')
            # print(f'C_times_C for pair={i}, l={l} is\n{C_times_C}')
            C_times_C = torch.outer(sto6g_coeff[qn_int[ni[i]]-1,l,:],sto6g_coeff[qn_int[nj[i]]-1,l,:])

            alpha_i = sto6g_exponent[qn_int[ni[i]]-1,l,:]*(zetas[idxi[i]]**2)
            alpha_j = sto6g_exponent[qn_int[nj[i]]-1,l,:]*(zetas[idxj[i]]**2)
            # alpha_i*alpha_j/(alpha_i+alpha_j)
            alphas_1 = (alpha_i[:, None]*alpha_j)/(alpha_i[:,None]+alpha_j)

            # <sA|sB>ij
            sij = ((2*torch.div(torch.sqrt(alpha_i[:, None]*alpha_j),alpha_i[:,None]+alpha_j))**(3/2))*torch.exp(-alphas_1*(rij[i]**2))

            # d/dx of <sA|sB>
            if l==0: 
                ans = 2.0*alphas_1*sij
                # TODO: check sign of Xij (i to j or j to i)? assuming Xij=Xj-Xi
                # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
                overlap_ab_x[i,0,0,0] = torch.sum(C_times_C*ans)*Xij[i,0]/a0_sq 
                overlap_ab_x[i,1,0,0] = torch.sum(C_times_C*ans)*Xij[i,1]/a0_sq 
                overlap_ab_x[i,2,0,0] = torch.sum(C_times_C*ans)*Xij[i,2]/a0_sq 

    overlap_ab_x[:,0,0,0] *= (beta[idxi,0]+beta[idxj,0])/2.0
    overlap_ab_x[:,1,0,0] *= (beta[idxi,0]+beta[idxj,0])/2.0
    overlap_ab_x[:,2,0,0] *= (beta[idxi,0]+beta[idxj,0])/2.0

    torch.set_printoptions(precision=6)
    # Core-core repulsion derivatives

    # First, derivative of g_AB
    tore = const.tore
    alpha = parnuc[0]
    rija=rij*a0
    ZAZB = tore[ni]*tore[nj]

    # special case for N-H and O-H
    XH = ((ni==7) | (ni==8)) & (nj==1)

    t2 = torch.zeros(npairs,dtype=dtype,device=device)
    tmp = torch.exp(-alpha[idxi]*rija)
    t2[~XH] = tmp[~XH]
    t2[XH] = tmp[XH]*rija[XH]
    t3 = torch.exp(-alpha[idxj]*rija)
    g = 1.0+t2+t3

    tmp = torch.exp(-alpha[idxi]*rija)
    prefactor = alpha[idxi]
    prefactor[XH] = prefactor[XH]*rija[XH]-1.0
    t3 = alpha[idxj]*torch.exp(-alpha[idxj]*rija)
    coreTerm = ZAZB*gam/rija*(prefactor*tmp+t3)
    core_ab_x = torch.zeros((npairs,3),dtype=dtype, device=device)
    core_ab_x[:,0] = coreTerm*Xij[:,0] 
    core_ab_x[:,1] = coreTerm*Xij[:,1]
    core_ab_x[:,2] = coreTerm*Xij[:,2]

    # Next, derivative of (sasa|sbsb), which can be done while doing derivatives of other two-center repulsion integral

    # Two-center repulsion integral derivatives

    # enuc is not computed at this moment
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)
    rho_0=torch.zeros_like(Z,dtype=dtype)
    isH = Z==1  # Hydrogen
    isX = Z>2   # Heavy atom
    rho_0[isH] = 0.5*ev/gss[isH]
    rho_0[isX] = 0.5*ev/gss[isX]
    rho0a = rho_0[idxi]
    rho0b = rho_0[idxj]
    w_x  = torch.zeros(rij.shape[0],3,10,10,dtype=dtype, device=device)
    # riHH = ev/sqrt(r0[HH]**2+(rho0a[HH]+rho0b[HH])**2)
    # TODO: Ask why rij is in bohr. It should be in angstrom right? Ans: OpenMopac website seems to suggest using bohr as well 
    # for the 2-e integrals.
    # Dividing by a0^2
    # again assuming xij = xj-xi, and hence forgoing the minus sign

    ev_a02 = ev/a0_sq
    term = ev_a02*torch.pow(rij[HH]**2+(rho0a[HH]+rho0b[HH])**2,-1.5)
    # TODO: Combine the 3 statements into 1
    w_x[HH,0,0,0] = term*Xij[HH,0]
    w_x[HH,1,0,0] = term*Xij[HH,1]
    w_x[HH,2,0,0] = term*Xij[HH,2]

    # TODO: Derivatives of the rotation matrix

    # Assembly
    # P is currently in the shape of (nmol,4*molsize, 4*molsize)
    # I will reshape it to P0(nmol*molsize*molsize, 4, 4)
    P0 = P.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol*molsize*molsize,4,4)

    # print(P0[mask,...])
    # print(overlap_ab_x[:,0,:,:])
    # print((P0[mask,:,:]*overlap_ab_x[:,0,:,:]).sum(dim=(1,2)))
    overlapx = torch.empty(npairs,3)
    overlapx[:,0] = (P0[mask,:,:]*overlap_ab_x[:,0,:,:]).sum(dim=(1,2))
    overlapx[:,1] = (P0[mask,:,:]*overlap_ab_x[:,1,:,:]).sum(dim=(1,2))
    overlapx[:,2] = (P0[mask,:,:]*overlap_ab_x[:,2,:,:]).sum(dim=(1,2))

    grad.index_add_(0,idxi,overlapx)
    grad.index_add_(0,idxj,overlapx,alpha=-1.0)

    grad.index_add_(0,idxi,core_ab_x)
    grad.index_add_(0,idxj,core_ab_x,alpha=-1.0)

    # print(grad)

    # ZAZB*g*d/dx(sasa|sbsb)
    overlapx[:,0] = ZAZB*g*w_x[:,0,0,0]
    overlapx[:,1] = ZAZB*g*w_x[:,1,0,0]
    overlapx[:,2] = ZAZB*g*w_x[:,2,0,0]

    grad.index_add_(0,idxi,overlapx)
    grad.index_add_(0,idxj,overlapx,alpha=-1.0)

    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    sumKAB = torch.zeros(npairs,3,4,4,dtype=dtype, device=device)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5*P0[mask]
    # print(f'Vishikh shape of w_x is {w_x.shape} and that of w_x[...,0,ind[i],:] is {w_x[...,0,ind[0],:].shape}')
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            # TODO: I think this can be folded in with some reshaping instead of 3 separate statements for x,y,z
            sumKAB[...,0,i,j] = torch.sum(Pp*w_x[...,0,ind[i],:][...,:,ind[j]],dim=(1,2))
            sumKAB[...,1,i,j] = torch.sum(Pp*w_x[...,1,ind[i],:][...,:,ind[j]],dim=(1,2))
            sumKAB[...,2,i,j] = torch.sum(Pp*w_x[...,2,ind[i],:][...,:,ind[j]],dim=(1,2))

    overlapx[:,0] = (P0[mask,:,:]*sumKAB[:,0,:,:]).sum(dim=(1,2))
    overlapx[:,1] = (P0[mask,:,:]*sumKAB[:,1,:,:]).sum(dim=(1,2))
    overlapx[:,2] = (P0[mask,:,:]*sumKAB[:,2,:,:]).sum(dim=(1,2))

    grad.index_add_(0,idxi,overlapx)
    grad.index_add_(0,idxj,overlapx,alpha=-1.0)

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

    #take out the upper triangle part in the same order as in W
    #shape (nparis, 10)

    PA = (P0[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
    PB = (P0[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))
    # print(f'Vishikh PA\n{PA}')
    # print(f'Vishikh PB\n{PB}')
    #suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    #suma shape (npairs, 10)
    suma_x = torch.sum(PA*w_x[:,0,:,:],dim=1)
    suma_y = torch.sum(PA*w_x[:,1,:,:],dim=1)
    suma_z = torch.sum(PA*w_x[:,2,:,:],dim=1)
    #sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    #sumb shape (npairs, 10)
    sumb_x = torch.sum(PB*w_x[:,0,:,:],dim=2)
    sumb_y = torch.sum(PB*w_x[:,1,:,:],dim=2)
    sumb_z = torch.sum(PB*w_x[:,2,:,:],dim=2)
    #print('suma:\n',suma)
    #reshape back to (npairs 4,4)
    # as will use index add in the following part
    sumA = torch.zeros(npairs,3,4,4,dtype=dtype, device=device)
    sumB = torch.zeros_like(sumA)
    sumA[...,0,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma_x
    sumA[...,1,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma_y
    sumA[...,2,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma_z
    sumB[...,0,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb_x
    sumB[...,1,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb_y
    sumB[...,2,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb_z

    overlapx[:,0] = (P0[maskd[idxj],:,:]*sumA[:,0,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*sumB[:,0,:,:]).sum(dim=(1,2))
    overlapx[:,1] = (P0[maskd[idxj],:,:]*sumA[:,1,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*sumB[:,1,:,:]).sum(dim=(1,2))
    overlapx[:,2] = (P0[maskd[idxj],:,:]*sumA[:,2,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*sumB[:,2,:,:]).sum(dim=(1,2))

    grad.index_add_(0,idxi,overlapx,alpha=0.5)
    grad.index_add_(0,idxj,overlapx,alpha=-0.5)


    '''
    #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
    #\sum_B
    F.index_add_(0,maskd[idxi],sumB)
    #\sum_A
    F.index_add_(0,maskd[idxj],sumA)
    '''

    # Core-elecron interaction
    e1b_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    e2a_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    e1b_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]
    e2a_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]

    overlapx[:,0] = (P0[maskd[idxj],:,:]*e2a_x[:,0,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,0,:,:]).sum(dim=(1,2))
    overlapx[:,1] = (P0[maskd[idxj],:,:]*e2a_x[:,1,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,1,:,:]).sum(dim=(1,2))
    overlapx[:,2] = (P0[maskd[idxj],:,:]*e2a_x[:,2,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,2,:,:]).sum(dim=(1,2))

    grad.index_add_(0,idxi,overlapx)
    grad.index_add_(0,idxj,overlapx,alpha=-1.0)

    # grad = grad.reshape(nmol,molsize,3)
    print(f'Analytical gradient is:\n{grad.view(nmol,molsize,3)}')

    if torch.any(isX):
        raise Exception("Analytical gradients not yet implemented for molecules with non-Hydrogen atoms")
