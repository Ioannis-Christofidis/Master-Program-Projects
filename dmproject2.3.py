import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as S
from scipy import stats
month_len=111.0
T=300#K
P=250*100
R=287.058
rho=P/(R*T)
#create arrays
a=S.Dataset("C:\Users\John\Desktop\master mathimata\winter semester\DYNAMICAL METEOROLOGY\project2-meteorology\mean sea level pressure 40 years.nc",mode='r')
b=S.Dataset("C:\Users\John\Desktop\master mathimata\winter semester\DYNAMICAL METEOROLOGY\project2-meteorology\merdional velocity 250hPa 40 years.nc",mode='r')
c=S.Dataset("C:\Users\John\Desktop\master mathimata\winter semester\DYNAMICAL METEOROLOGY\project2-meteorology\u and v 550 hPa at 40 years.nc",mode='r')
#d=S.Dataset("C:\Users\Johnlaptop\Videos\Desktop\project2-meteorology\mv550.nc",mode='r')
lat=a.variables["latitude"][:]
lon=a.variables["longitude"][:]
time = a.variables["time"][:]
time1 = month_len * (time - np.min(time)) / ( np.max(time) - np.min(time))
mlsp=a.variables["msl"][:]
v250=b.variables["v"][:,:,:]
u550=c.variables["u"][:,:,:]
v550=c.variables["v"][:,:,:]
pressure_av=np.mean(mlsp,axis=2)
pressure30=pressure_av[:,30]
pressure0=pressure_av[:,0]
pres_dif=pressure30-pressure0
v250_av=np.mean(v250,axis=2)
meridonalflux=rho*v250_av[:,15]
u550_av=np.mean(u550,axis=2)
v550_av=np.mean(v550,axis=2)
v550_15=v550_av[:,15]
u550_15=u550_av[:,15]
u_v=u550_15-v550_15
#plt.scatter(pres_dif,meridonalflux)
#plt.show()
a=np.corrcoef(pres_dif,meridonalflux)[0,1]
b=np.corrcoef(pres_dif,u_v)[0,1]
c=np.corrcoef(meridonalflux,u_v)[0,1]
f=[[1.,a,b],[a,1.,c],[b,c,1.]]
l,E=np.linalg.eigh(f)
S=sum(l)
ldev=l/S

#mean pressure deviation
mpd=np.mean(pres_dif)
sd=np.std(pres_dif)
pd_dev =(mpd-pres_dif)/sd
#mean meridional flux deviation
mmf=np.mean(meridonalflux)
sdmf=np.std(meridonalflux)
mf_dev=(mmf-meridonalflux)/sdmf
#u-v deviation
mu_v=np.mean(u_v)
sduv=np.std(u_v)
u_vdev=(mu_v-u_v)/sduv
x=[pd_dev,mf_dev,u_vdev]
lmax=max(l)
lmin=min(l)
for i in range(0,3):
    if lmax==l[i]:
        i_max=i
    elif lmin==l[i]:
        i_min=i
    else:
        i_med=i
        
Y=np.dot(E,x)
Y1=Y[i_max,:]
Y2=Y[i_med,:]
Y3=Y[i_min,:]
plt.figure(1)
time2=time1/3+1978+2/3.
d=np.corrcoef(time2,meridonalflux)[0,1]
plt.scatter(time2,Y1)
plt.xlim(1978,2020)
plt.xlabel('Time (years)')
plt.ylabel('Y1component')
plt.figure(2)
plt.scatter(time2,Y2)
plt.xlim(1978,2020)
plt.xlabel('Time (years)')
plt.ylabel('Y2component')
plt.figure(3)
plt.scatter(time2,Y3)
plt.xlim(1978,2020)
plt.xlabel('Time (years)')
plt.ylabel('Y3component')
plt.figure(4)
plt.scatter(u_v,meridonalflux)
plt.xlabel('U-V(m/s)')
plt.ylabel('Meridional Flux(kg/s)')
fit1=np.polyfit(u_v,meridonalflux,deg=1)
plt.plot(u_v,fit1[0]*u_v+fit1[1],label='r=%f' %c)
plt.legend()
fig=plt.figure(5)
plt.scatter(u_v,pres_dif)
plt.xlabel('U-V(m/s)')
plt.ylabel('Pressure difference(hPa)')
fit2=np.polyfit(u_v,pres_dif,deg=1)
plt.plot(u_v,fit2[0]*u_v+fit2[1],label='r=%f' %b)
plt.legend()
plt.figure(6)
plt.scatter(meridonalflux,pres_dif)
plt.xlabel('Meridonal flux(Kg/s)')
plt.ylabel('Pressure difference(hPa)')
fit3=np.polyfit(meridonalflux,pres_dif,deg=1)
plt.plot(meridonalflux,fit3[0]*meridonalflux+fit3[1],label='r=%f' %a)
plt.legend()
plt.figure(7)
plt.scatter(time2,meridonalflux)
plt.xlabel('Time (years)')
plt.ylabel('Meridionalflux (Kg/s)')
plt.xlim(1978,2020)
plt.figure(8)
plt.scatter(time2,pres_dif)
plt.xlabel('Time (years)')
plt.ylabel('Pressure difference (hPa)')
plt.xlim(1978,2020)
fit3=np.polyfit(time2,meridonalflux,deg=1)
plt.plot(time2,fit3[0]*time2+fit3[1],label='r=%f' %d)
plt.figure(9)
plt.scatter(time2,u_v)
plt.xlabel('Time (years)')
plt.ylabel('U-V (m/s)')
plt.xlim(1978,2020)
fit3=np.polyfit(time2,u_v,deg=1)
plt.plot(time2,fit3[0]*time2+fit3[1],label='r=%f' %d)
plt.legend()
plt.figure(10)
plt.scatter(time2,Y1,color='blue',label='Y1 component')
plt.scatter(time2,Y2,color='green',label='Y2 component')
plt.scatter(time2,Y3,color='red',label='Y3 component')
plt.ylabel('Y component')
plt.xlabel('Time (years)')
plt.xlim(1978,2020)
plt.legend()
plt.show()
