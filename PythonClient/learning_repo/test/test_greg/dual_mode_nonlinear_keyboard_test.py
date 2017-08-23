#!/usr/bin/env python
""" nonlinear_control_keyboard_test.py:
Custom control of quadrotor with keyboard using full nonlinear dynamics
"""

__author__ = "Vinicius Guimaraes Goecks and Gregory Michael Gremillion"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 7, 2017"

# import
from PythonClient import *
import sys
import time
from math import cos, sin, tan, atan2, pi
import pygame, time           # for keyboard multi-threaded inputs
from pygame.locals import *
import datetime
import numpy as np

def getKeyboardCommands(stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):
    pygame.event.get()

    # control
    K_alt = .25
    drll = my_joystick.get_axis(3) # 0 = right stick lat
    dpch = my_joystick.get_axis(4) # 0 = right stick long
    dyaw = my_joystick.get_axis(0) # 0 = left stick lat
    dthr = K_alt*my_joystick.get_axis(1) # 0 = left stick long

    # options
    if my_joystick.get_button(6): # select button
        mode = 0

    if my_joystick.get_button(7): # start button
        mode = 1

    if  my_joystick.get_button(8): # big xbox button
        rset = True # reset position/controls
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)

    # flush rest and display
    pygame.event.clear()

    rll = drll / 4.0
    pch = dpch / 4.0
    yaw = dyaw / 1.0
    thr += dthr / 20.0


    return (stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

def getKeyboardCommands2(stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):
    for event in pygame.event.get():
        if event.type == pygame.KEYUP:
            # LATERAL COMMANDS
            if event.key == pygame.K_RIGHT:
                drll = 0.0
            if event.key == pygame.K_LEFT:
                drll = 0.0
            # LONGITUDINAL COMMANDS
            if event.key == pygame.K_UP:
                dpch = 0.0 # speed forward
            if event.key == pygame.K_DOWN:
                dpch = 0.0 # break drone
            # HEAVE COMMANDS
            if event.key == pygame.K_PAGEUP:
                dthr = 0.0 # heave up
            if event.key == pygame.K_PAGEDOWN:
                dthr = 0.0 # heave down
            # HEAVE COMMANDS
            if event.key == pygame.K_a:
                dyaw = 0.0 # heave up
            if event.key == pygame.K_d:
                dyaw = 0.0 # heave down
            if event.key == pygame.K_m:
                mode += 1
                if mode > 1:
                    mode = 0
        if event.type == pygame.KEYDOWN:
            # LATERAL COMMANDS
            if event.key == pygame.K_RIGHT:
                drll = 1.0
            if event.key == pygame.K_LEFT:
                drll = -1.0
            # LONGITUDINAL COMMANDS
            if event.key == pygame.K_UP:
                dpch = -1.0 # speed forward
            if event.key == pygame.K_DOWN:
                dpch = 1.0 # break drone
            # HEAVE COMMANDS
            if event.key == pygame.K_PAGEUP:
                dthr = -1.0 # heave up
            if event.key == pygame.K_PAGEDOWN:
                dthr = 1.0 # heave down
            # HEAVE COMMANDS
            if event.key == pygame.K_a:
                dyaw = -1.0 # yaw left
            if event.key == pygame.K_d:
                dyaw = 1.0 # yaw right
            if event.key == pygame.K_DELETE:
                stat = False # break drone
            if event.key == pygame.K_SPACE:
                rset = True # reset position/controls
                rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
    rll += drll / 100.0
    pch += dpch / 100.0
    yaw += dyaw / 100.0
    thr += dthr / 100.0

    return (stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

# assign client
client = AirSimClient('127.0.0.1')
control = 'joystick' # 'keyboard'

# initialize pygame for inputs
if control == 'keyboard':
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((500, 120))
    pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')

elif control == 'joystick':
    pygame.init()
    my_joystick = pygame.joystick.Joystick(0)
    my_joystick.init()


stat = True
mode = 0 # 0 = independent axis control, 1 = forward velocity and coordinated turn control
V_ref = 0

pos = client.getPosition()
orq = client.getOrientation()
pos0,orq0 = (pos,orq)
pos_new,orq_new = (pos,orq)
ore0 = client.toEulerianAngle(orq)

rll_prv,pch_prv,yaw_prv,thr_prv = (0.0,0.0,0.0,0.0)
drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
drll_prv,dpch_prv,dyaw_prv,dthr_prv = (0.0,0.0,0.0,0.0)
mode_prv = mode

orq_new = [0.0,0.0,0.0,0.0]
ore_new = [0.0,0.0,0.0]
ore_check = [0.0,0.0,0.0]

ps = 0

g = 9.81 # gravity [m/s^2]
m = 0.6 # mass [kg]
Ix,Iy,Iz = (0.00115,0.00115,0.00598) # inertia tensor principle components [kg m^2] (based on Gremillion 2016 https://arc.aiaa.org/doi/abs/10.2514/1.J054408)
Lp,Mq,Nr,Xu,Yv,Zw = (-0.01,-0.01,-0.2,-0.05,-0.05,-0.05) # aerodynamic drag derivatives [kg / s]

K_ph,K_p = (0.5,-0.02)
K_th,K_q = (K_ph,K_p)
K_r,K_dr = (0.2,0.0)
K_dps = 0.2
K_z,K_dz,K_z_i = (-20.0,5.0,-0.05)
K_dv = 0.5
K_v = 1.0

# initialize states
ph,th,ps,p,q,r,u,v,w,x,y,z = (ore0[0],ore0[1],ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,pos0[0],pos0[1],pos0[2])
# initialize state derivatives
ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
# initialize state derivatives
dph,dth,dps,dp,dq,dr,du,dv,dw,dx,dy,dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
# initialize aerodynamic states
tawx,tawy,tawz,fwx,fwy,fwz = (0.0,0.0,0.0,0.0,0.0,0.0) # wind forces [N] torques [N m]

z_i = 0.0 # z error integral

original_dt = 0.02
previous_time = time.time() + original_dt
while stat:
    current_time = time.time()
    dt = current_time - previous_time
    # print('dt: ', dt)
    dt = np.clip(dt,0,original_dt)

    rset = False
    mode = mode_prv

    # retrieve states from buffer
    ph,th,ps,p,q,r,u,v,w,x,y,z = (ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv)

    # retrieve commands from buffer
    rll,pch,yaw,thr = (rll_prv,pch_prv,yaw_prv,thr_prv)
    drll,dpch,dyaw,dthr = (drll_prv,dpch_prv,dyaw_prv,dthr_prv)

    # get commands from keyboard
    stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr = getKeyboardCommands(stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

    # zero commands if mode changes
    if mode != mode_prv:
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        tawx,tawy,tawz,fwx,fwy,fwz = (0.0,0.0,0.0,0.0,0.0,0.0)
        ph,th,p,q,r,u,v,w = (ore0[0],ore0[1],0.0,0.0,0.0,0.0,0.0,0.0)
        dph,dth,dps,dp,dq,dr,du,dv,dw,dx,dy,dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        be = 0.0

    # wind angles
    al = atan2(w,u)
    be = atan2(v,u)

    # control mode case
    if mode == 0:
        # state reference
        ph_ref = 2.0 * rll
        th_ref = 2.0 * pch
        dps_ref = 10.0 * yaw

        # state error
        ph_err = ph_ref - ph
        th_err = th_ref - th
        dps_err = dps_ref - dps

        # force/torque control
        tax = K_ph * ph_err + K_p * p
        tay = 3.0 * K_th * th_err + 4.0 * K_q * q
        taz = K_dps * dps_err
    else:
        # state reference
        ph_ref = 2.0 * rll
        V_ref += -1.0 * pch
        dps_ref = 10.0 * be
        V = u * cos(th) + w * sin(th)
        V_err = V_ref - V
        th_ref = -V_err

        # state error
        ph_err = ph_ref - ph
        th_err = th_ref - th
        dps_err = dps_ref - dps

        # force/torque control
        tax = K_ph * ph_err + K_p * p
        tay = 3.0 * K_th * th_err + 4.0 * K_q * q
        if V <=0.1:
            taz = 0
        else:
            taz = K_dps * dps_err

    # heave control
    z_ref = 10.0 * thr + pos0[2]
    z_err = z_ref - z
    ft = K_z * z_err + K_dz * dz + K_z_i * z_i + m * g

    # drag terms
    tawx = Lp * p
    tawy = Mq * q
    tawz = Nr * r
    fwx = Xu * u
    fwy = Yv * v
    fwz = Zw * w

    # nonlinear dynamics
    dph = p + r * (cos(ph) * tan(th)) + q * (sin(ph) * tan(th))
    dth = q * (cos(ph)) - r * (sin(ph))
    dps = r * (cos(ph) / cos(th)) + q * (sin(ph) / cos(th))
    dp = r * q * (Iy - Iz) / Ix + (tax + tawx) / Ix
    dq = p * r * (Iz - Ix) / Iy + (tay + tawy) / Iy
    dr = p * q * (Ix - Iy) / Iz + (taz + tawz) / Iz
    du = r * v - q * w - g * (sin(th)) + fwx / m
    dv = p * w - r * u + g * (sin(ph) * cos(th)) + fwy / m
    dw = q * u - p * v + g * (cos(th) * cos(ph)) + (fwz - ft) / m
    dx = w * (sin(ph) * sin(ps) + cos(ph) * cos(ps) * sin(th)) - v * (cos(ph) * sin(ps) - cos(ps) * sin(ph) * sin(th)) + u * (cos(ps) * cos(th))
    dy = v * (cos(ph) * cos(ps) + sin(ph) * sin(ps) * sin(th)) - w * (cos(ps) * sin(ph) - cos(ph) * sin(ps) * sin(th)) + u * (cos(th) * sin(ps))
    dz = w * (cos(ph) * cos(th)) - u * (sin(th)) + v * (cos(th) * sin(ph))

    # numerically integrate states
    ph += dph * dt
    th += dth * dt
    ps += dps * dt
    p += dp * dt
    q += dq * dt
    r += dr * dt
    u += du * dt
    v += dv * dt
    w += dw * dt
    x += dx * dt
    y += dy * dt
    z += dz * dt

    ## GENERATE POSITION/ORIENTATION VISUALIZATION
    # update position/orientation
    ore_new = [ph,th,ps] # new orientation (Euler)
    orq_new = client.toQuaternion(ore_new) # convert to quaternion
    pos_new = [x,y,z] # new position

    # convert back to Euler (check quaternion conversion)
    ore_check = client.toEulerianAngle(orq_new)

    if rset == True:
        client.simSetPose(pos0,orq0) # reset to origin
        pos_new,orq_new = (pos0,orq0) # update current position/orientation to origin
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        ph,th,ps,p,q,r,u,v,w,x,y,z = (ore0[0],ore0[1],ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,pos0[0],pos0[1],pos0[2])
        ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
        z_i = 0.0
        V_ref = 0

    # get vehicle status
    pos = client.getPosition() # get current position
    orq = client.getOrientation() # get current orientation
    collision = client.getCollisionInfo() # get collision status

    if ((collision[0] == True) or (ore_new[0] > .8*np.pi/2) or (ore_new[1] > .8*np.pi/2)): # if collision, reset position/orientation/controls
        print("COLLISION - Resetting")
        client.simSetPose(pos0,orq0) # reset to origin
        pos_new,orq_new = (pos0,orq0) # update current position/orientation to origin
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        ph,th,ps,p,q,r,u,v,w,x,y,z = (ore0[0],ore0[1],ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,pos0[0],pos0[1],pos0[2])
        ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
        z_i = 0.0
        V_ref = 0
        
    else: # if no collision update position/orientation
        client.simSetPose(pos_new,orq_new)

    # z error integral
    z_i += z_err * dt

    # send states to buffer
    ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)

    # send commands to buffer
    rll_prv,pch_prv,yaw_prv,thr_prv = (rll,pch,yaw,thr)
    drll_prv,dpch_prv,dyaw_prv,dthr_prv = (drll,dpch,dyaw,dthr)
    mode_prv = mode

    # print debug output command/orientation
    # print("phr %f, dpsr %f, be %f, V %f, Vref %f, roll %f, phi %f, pitch %f, theta %f, yaw %f, r %f, throttle %f, z %f" %(ph_ref,dps_ref,be,V,V_ref,rll,ph,pch,th,yaw,r,thr,z))
    # print("mode %f, roll %f, pitch %f, yaw %f, throttle %f" %(mode,rll,pch,yaw,thr))

    previous_time = current_time
