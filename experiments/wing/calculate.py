import math
n = 64
f = 24
lamb = 126
SelExt_Orca_Online = 2*(n-f)+1
SelExt_TruncXpert_Online = n-f+1
TruncReLU_Orca_Online = 2*n-f+5
TruncReLU_TruncXpert_Online = n-f+1

SelExt_Orca_Offline = 6*n+(n-f)*(lamb+3)+lamb+2
SelExt_TruncXpert_Offline = 6*n-f+1
TruncReLU_Orca_Offline = 7*n+2*lamb-f+n*(lamb+3)
TruncReLU_TruncXpert_Offline = 6*n-f+2+2*lamb+(n-f-1+math.ceil(math.log2(lamb+1)))*(lamb+2)
print((n-f-1+math.log2(lamb+1)))
print(math.ceil(math.log2(lamb+1)))
print(SelExt_Orca_Online/SelExt_TruncXpert_Online, TruncReLU_Orca_Online/TruncReLU_TruncXpert_Online)

print(SelExt_Orca_Offline/SelExt_TruncXpert_Offline, TruncReLU_Orca_Offline/TruncReLU_TruncXpert_Offline)