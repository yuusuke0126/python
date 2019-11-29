

vx = 0.7
wz = 3.0

bias = 0.4
kcTireTread_c = 435
kcMaxSpeedMMPM_c = 500
kcSpeedLimitMPM_c = 49
kcMinRadius_c = 500
kcMaxMPMForCurve_c = 42

radius = vx * 1000.0 / wz
speed = (radius + 1000.0 * bias / 2.0) * wz

radius = int(radius)
speed = int(speed)
print(speed, radius)

v = int(abs(speed)) * 60
r = int(abs(radius))
if v > kcSpeedLimitMPM_c * 1000:
  v = kcSpeedLimitMPM_c * 1000

if v > kcMaxMPMForCurve_c * 1000:
  v = kcMaxMPMForCurve_c * 1000

if r < kcMinRadius_c:
  r = kcMinRadius_c

print(v, r)

if 1 < radius:
  vr = v * (r + kcTireTread_c / 2) / r
  vl = v * (r - kcTireTread_c / 2) / r
  vd = vr - vl
  vr = v
  vl = v - vd
else:
  vr = v * (r - kcTireTread_c / 2) / r
  vl = v * (r + kcTireTread_c / 2) / r
  vd = vl - vr
  vl = v
  vr = v - vd

vr_value = int( (( vr * 10 * 255 ) + ( kcMaxSpeedMMPM_c * 1000 / 2 )) / kcMaxSpeedMMPM_c / 1000 )
vl_value = int( (( vl * 10 * 255 ) + ( kcMaxSpeedMMPM_c * 1000 / 2 )) / kcMaxSpeedMMPM_c / 1000 )

print("vr = ",vr_value, ", vl = ", vl_value)