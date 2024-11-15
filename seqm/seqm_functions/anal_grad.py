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
    torch.set_printoptions(precision=6)
    # print("Reached gradient")
    # print(f'Vishikh: shape of P is {P.shape}')
    # print(f'Vishikh: P is {P}')

    dtype = Xij.dtype
    device = Xij.device
    nmol = P.shape[0]
    npairs=Xij.shape[0]
    qn_int = const.qn_int
    a0_sq = a0*a0
    # print(f'There are {nmol} molecules with {molsize} atoms in each molecule')
    # print(f'The convered density is \n{P} whose shape is {P.shape}')

    # Define the gradient tensor
    grad = torch.zeros(nmol*molsize, 3, dtype=dtype, device=device)
    # grad_nuc = torch.zeros(nmol*molsize, 3, dtype=dtype, device=device)

    # Overlap grad
    overlap_KAB_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    # Correct-ish so far (gaussian approximation of sto might be giving some difference)
    overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij)

    # Core-core repulsion derivatives

    # First, derivative of g_AB
    tore = const.tore
    alpha = parnuc[0]
    ZAZB = tore[ni]*tore[nj]
    pair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)
    g = core_core_der(alpha,rij,Xij,ZAZB,ni,nj,idxi,idxj,gam,pair_grad)
    # print(f'gam is\n{gam}\ng is\n{g}')

    # mycoreTerm = (1.0/rija)*(prefactor*tmp+t3)
    # mypair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)
    # mypair_grad[:,0] = mycoreTerm*Xij[:,0] 
    # mypair_grad[:,1] = mycoreTerm*Xij[:,1]
    # mypair_grad[:,2] = mycoreTerm*Xij[:,2]
    # print(mypair_grad)
    # # mycoreTerm is correct

    # Two-center repulsion integral derivatives
    # Core-valence integral derivatives e1b_x and e2a_x also calculated
    w_x  = torch.zeros(rij.shape[0],3,10,10,dtype=dtype, device=device)
    e1b_x,e2a_x = w_der(Z,tore,ni,nj,w_x,rij,Xij,idxi,idxj,gss,)

    # Assembly
    # P is currently in the shape of (nmol,4*molsize, 4*molsize)
    # I will reshape it to P0(nmol*molsize*molsize, 4, 4)
    P0 = P.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol*molsize*molsize,4,4)

    # print(P0[mask,...])
    # print(overlap_KAB_x[:,0,:,:])
    # print((P0[mask,:,:]*overlap_KAB_x[:,0,:,:]).sum(dim=(1,2)))
    overlapx = torch.empty(npairs,3)
    # overlapx[:,0] = (P0[mask,:,:]*overlap_KAB_x[:,0,:,:]).sum(dim=(1,2))
    # overlapx[:,1] = (P0[mask,:,:]*overlap_KAB_x[:,1,:,:]).sum(dim=(1,2))
    # overlapx[:,2] = (P0[mask,:,:]*overlap_KAB_x[:,2,:,:]).sum(dim=(1,2))
    # pair_grad.add_((P0[mask,None,:,:]*overlap_KAB_x).sum(dim=(2,3)))

    # grad.index_add_(0,idxi,overlapx)
    # grad.index_add_(0,idxj,overlapx,alpha=-1.0)


    # grad_nuc.index_add_(0,idxi,pair_grad)
    # grad_nuc.index_add_(0,idxj,pair_grad,alpha=-1.0)

    # print(grad)

    # ZAZB*g*d/dx(sasa|sbsb)
    pair_grad.add_((ZAZB*g).unsqueeze(1)*w_x[:,:,0,0])
    # overlapx[:,0] = ZAZB*g*w_x[:,0,0,0]
    # overlapx[:,1] = ZAZB*g*w_x[:,1,0,0]
    # overlapx[:,2] = ZAZB*g*w_x[:,2,0,0]
    #
    # grad.index_add_(0,idxi,overlapx)
    # grad.index_add_(0,idxj,overlapx,alpha=-1.0)

    # grad_nuc.index_add_(0,idxi,overlapx)
    # grad_nuc.index_add_(0,idxj,overlapx,alpha=-1.0)

    # off diagonal block part, check KAB in forck2.f
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
    # print(f'Vishikh shape of w_x is {w_x.shape} and that of w_x[...,0,ind[i],:] is {w_x[...,0,ind[0],:].shape}')
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[...,i,j] += torch.sum(Pp.unsqueeze(1)*(w_x[...,ind[i],:][...,:,ind[j]]),dim=(2,3))
            # TODO: I think this can be folded in with some reshaping instead of 3 separate statements for x,y,z
            # overlap_KAB_x[...,0,i,j] += torch.sum(Pp*w_x[...,0,ind[i],:][...,:,ind[j]],dim=(1,2))
            # overlap_KAB_x[...,1,i,j] += torch.sum(Pp*w_x[...,1,ind[i],:][...,:,ind[j]],dim=(1,2))
            # overlap_KAB_x[...,2,i,j] += torch.sum(Pp*w_x[...,2,ind[i],:][...,:,ind[j]],dim=(1,2))

    pair_grad.add_((P0[mask,None,:,:]*overlap_KAB_x).sum(dim=(2,3)))
    # overlapx[:,0] = (P0[mask,:,:]*sumKAB[:,0,:,:]).sum(dim=(1,2))
    # overlapx[:,1] = (P0[mask,:,:]*sumKAB[:,1,:,:]).sum(dim=(1,2))
    # overlapx[:,2] = (P0[mask,:,:]*sumKAB[:,2,:,:]).sum(dim=(1,2))

    # grad.index_add_(0,idxi,pair_grad)
    # grad.index_add_(0,idxj,pair_grad,alpha=-1.0)
    # grad.index_add_(0,idxi,overlapx)
    # grad.index_add_(0,idxj,overlapx,alpha=-1.0)

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
    
    # # Compute suma and sumb for each dimension x, y, z efficiently
    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)

    # Initialize sumA and sumB tensors
    # sumA = torch.zeros(npairs, 3, 4, 4, dtype=dtype, device=device)
    # sumB = torch.zeros_like(sumA)
    # reususe overlap_KAB_x here instead of creating a new array

    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    pair_grad.add_(0.5*torch.sum(P0[maskd[idxj]].unsqueeze(1) * sumA, dim=(2, 3)))
    sumB = overlap_KAB_x
    sumB.zero_()
    sumB[..., indices[0], indices[1]] = sumb
    pair_grad.add_(0.5*torch.sum(P0[maskd[idxi]].unsqueeze(1) * sumB, dim=(2, 3)))

    # Compute overlapx using broadcasting and summing
    # overlapx = torch.sum(P0[maskd[idxj]].unsqueeze(1) * sumA, dim=(2, 3)) + torch.sum(P0[maskd[idxi]].unsqueeze(1) * sumB, dim=(2, 3))
    # grad.index_add_(0,idxi,overlapx,alpha=0.5)
    # grad.index_add_(0,idxj,overlapx,alpha=-0.5)

    # pair_grad.add_(0.5*(torch.sum(P0[maskd[idxj]].unsqueeze(1) * sumA, dim=(2, 3)) + torch.sum(P0[maskd[idxi]].unsqueeze(1) * sumB, dim=(2, 3))))


    # Core-elecron interaction
    pair_grad.add_((P0[maskd[idxj],None,:,:]*e2a_x).sum(dim=(2,3)) + (P0[maskd[idxi],None,:,:]*e1b_x).sum(dim=(2,3)))
    # overlapx[:,0] = (P0[maskd[idxj],:,:]*e2a_x[:,0,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,0,:,:]).sum(dim=(1,2))
    # overlapx[:,1] = (P0[maskd[idxj],:,:]*e2a_x[:,1,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,1,:,:]).sum(dim=(1,2))
    # overlapx[:,2] = (P0[maskd[idxj],:,:]*e2a_x[:,2,:,:]).sum(dim=(1,2)) + (P0[maskd[idxi],:,:]*e1b_x[:,2,:,:]).sum(dim=(1,2))
    #
    # grad.index_add_(0,idxi,overlapx)
    # grad.index_add_(0,idxj,overlapx,alpha=-1.0)
    grad.index_add_(0,idxi,pair_grad)
    grad.index_add_(0,idxj,pair_grad,alpha=-1.0)


    # grad = grad.reshape(nmol,molsize,3)
    print(f'Analytical gradient is:\n{grad.view(nmol,molsize,3)}')
    # print(f'Analytical gradient of Enuc is:\n{grad_nuc.view(nmol,molsize,3)}')

def overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij):
    a0_sq = a0*a0
    # s orbitals
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
    # print(f'Vishikh: {torch.sum(C_times_C*sij)}')
    # ss Overlap matrix is correct

    # d/dx of <sA|sB>
    ans = 2.0*alphas_1*sij
    # TODO: check sign of Xij (i to j or j to i)? assuming Xij=Xj-Xi
    # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
    overlap_KAB_x[:,:,0,0] = ((beta[idxi,0]+beta[idxj,0])*torch.sum(C_times_C*ans,dim=(1,2))).unsqueeze(1)*Xij[:,:]/a0_sq 
        
    # There is no dividing beta_mu+beta_nu by 2. Found this out during debugging.
    # Possibly because here we're only going over unique pairs, but in the total energy
    # expression the overlap term appears on the upper and lower triangle of Hcore 
    # and hence needs to be multiplied by 2.
    # overlap_KAB_x[:,0,0,0] *= (beta[idxi,0]+beta[idxj,0])#/2.0
    # overlap_KAB_x[:,1,0,0] *= (beta[idxi,0]+beta[idxj,0])#/2.0
    # overlap_KAB_x[:,2,0,0] *= (beta[idxi,0]+beta[idxj,0])#/2.0

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
    # pair_grad[:,0] = coreTerm*Xij[:,0] 
    # pair_grad[:,1] = coreTerm*Xij[:,1]
    # pair_grad[:,2] = coreTerm*Xij[:,2]
    return g

def w_der(Z,tore,ni,nj,w_x,rij,Xij,idxi,idxj,gss,):
    # Two-center repulsion integral derivatives
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)
    rho_0=torch.zeros_like(Z,dtype=rij.dtype)
    isH = Z==1  # Hydrogen
    isX = Z>2   # Heavy atom
    rho_0[isH] = 0.5*ev/gss[isH]
    rho_0[isX] = 0.5*ev/gss[isX]
    rho0a = rho_0[idxi]
    rho0b = rho_0[idxj]

    # d/dx (ss|ss)
    # (ss|ss) = riHH = ev/sqrt(r0[HH]**2+(rho0a[HH]+rho0b[HH])**2)
    # Why is rij in bohr? It should be in angstrom right? Ans: OpenMopac website seems to suggest using bohr as well 
    # for the 2-e integrals.
    # Dividing by a0^2 for gradient in eV/ang
    # again assuming xij = xj-xi, and hence forgoing the minus sign

    ev_a02 = ev/a0/a0
    term_ss = ev_a02*torch.pow(rij[HH]**2+(rho0a[HH]+rho0b[HH])**2,-1.5)
    w_x[HH,:,0,0] = term_ss.unsqueeze(1)*Xij[HH,:]
    # w_x[HH,0,0,0] = term_ss*Xij[HH,0]
    # w_x[HH,1,0,0] = term_ss*Xij[HH,1]
    # w_x[HH,2,0,0] = term_ss*Xij[HH,2]

    # TODO: Derivatives of the rotation matrix
    if torch.any(isX):
        raise Exception("Analytical gradients not yet implemented for molecules with non-Hydrogen atoms")

    # Core-elecron interaction
    e1b_x = torch.zeros((rij.shape[0],3,4,4),dtype=w_x.dtype, device=w_x.device)
    e2a_x = torch.zeros((rij.shape[0],3,4,4),dtype=w_x.dtype, device=w_x.device)
    e1b_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]
    e2a_x[HH,:,0,0] = -tore[1]*w_x[HH,:,0,0]
    return e1b_x,e2a_x
