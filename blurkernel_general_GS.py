import torch
import torch.nn as nn



def get_gaussian_kernel(kernel_size, sigma1,sigma2,rho,dtype):
   
    gaussian_kernel = torch.zeros(int(kernel_size), int(kernel_size)).to('cuda')
    radias = kernel_size // 2

    for y in range(-radias, radias + 1):
        for x in range(-radias, radias + 1):
            gaussian_kernel[y + radias, x + radias] = torch.exp(-x ** 2/(2.0 * sigma1 ** 2)-2*rho*x*y/(2*sigma1*sigma2)-y ** 2 / (2.0 * sigma2 ** 2))

  
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel
##








##
class Blurkernel(nn.Module):
    def __init__(self,kernel_size,data_type):
        super(Blurkernel, self).__init__()
        self.kernel_size=kernel_size
        # print(' self.kernel_size:', self.kernel_size)
        self.sigma1 = nn.Parameter(torch.rand(1)*3, requires_grad=True)
        self.sigma2= nn.Parameter(torch.rand(1)*3, requires_grad=True)
        # self.rho=nn.Parameter(torch.rand(1)*0.5-0.3, requires_grad=True)
        self.rho=nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.data_type=data_type

    def forward(self):

        # print('<<<<<<<<self.sigma1,sigma2 rho:',self.sigma1,self.sigma2,self.rho)
   
        gaussian_kernel = get_gaussian_kernel(self.kernel_size, self.sigma1, self.sigma2,self.rho,self.data_type)

        kernel_raw=gaussian_kernel.unsqueeze(0).unsqueeze(0)

        blur_kernel = kernel_raw / kernel_raw.sum()
        return blur_kernel


