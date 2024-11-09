import torch
from .constants import sto6g_coeff, sto6g_exponent

def scf_grad(P, const, molsize, idxi,idxj, ni,nj,xij,rij,
            Z, zetas,zetap):
    """
    Calculate the gradient of the ground state SCF energy
    """
    print("Reached gradient")

    dtype = xij.dtype
    device = xij.device
    nmol = P.shape[0]
    npairs=xij.shape[0]
    qn_int = const.qn_int
    print(f'There are {nmol} molecules with {molsize} atoms in each molecule')
    # print(f'The convered density is \n{P} whose shape is {P.shape}')

    # Define the gradient tensor
    grad = torch.zeros(nmol, 3, molsize, dtype=dtype, device=device)

    # Overlap grad
    # TODO: Add a cutoff distance for overlap
    overlap_ab_x = torch.zeros((npairs,3,4,4),dtype=dtype, device=device)
    # overlap_ab_x([pair_number], [x_der,y_der,z_der],[sA,pxA,pyA,pzA],[sB,pxB,pyB,pzB])

    print(f'zeta_s is\n{zetas}')
    print(f'Principal quantum no of ni is\n{qn_int[ni]}')
    print(f'Principal quantum no of nj is\n{qn_int[nj]}')
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
                ans = -2.0*alphas_1*sij


