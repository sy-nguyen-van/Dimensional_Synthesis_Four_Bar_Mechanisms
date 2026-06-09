import math
import cmath

x = [14.201369, 8.0143127, 16.309922, 12.253871, 27.797721, 11.305412, -2.3095310, 45.695083, 4.5172417, 6.2532447, 2.4038981, 2.9362397, 3.4905751, 4.0650776, 5.2095019]
Cdx=[20, 20, 20, 20, 20, 20]
Cdy=[20, 25, 30, 35, 40, 45]

r1=x[0]
r2=x[1]
r3=x[2]
r4=x[3]
rcx=x[4]
rcy=x[5]
x7=x[6]
x8=x[7]
x9=x[8]

f = 0
for i in range(6):
    teta2 = x[9+i]
    K1 = r1/r2
    K4 = r1/r3
    K5 = (r4**2 - r1**2 - r2**2 - r3**2) / (2*r3*r2)
    D = math.cos(teta2) - K1 + K4*math.cos(teta2) + K5
    E = -2*math.sin(teta2)
    F_val = K1 + (K4-1)*math.cos(teta2) + K5
    
    disc = E**2 - 4*D*F_val
    ATAN3 = (-E - cmath.sqrt(disc)) / (2*D)
    
    teta3 = (2*cmath.atan(ATAN3)).real
    
    Crx = r2*math.cos(teta2) + rcx*math.cos(teta3) - rcy*math.sin(teta3)
    Cry = r2*math.sin(teta2) + rcx*math.sin(teta3) + rcy*math.cos(teta3)
    
    C_x = math.cos(x9)*Crx - math.sin(x9)*Cry + x7
    C_y = math.sin(x9)*Crx + math.cos(x9)*Cry + x8
    
    err = (C_x - Cdx[i])**2 + (C_y - Cdy[i])**2
    f += err
    print(f"Point {i}: error {err}, teta2={teta2}, disc={disc}")

print("Total Cost:", f)
