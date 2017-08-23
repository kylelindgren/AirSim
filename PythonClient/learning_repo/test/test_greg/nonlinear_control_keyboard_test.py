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
from math import cos, sin, tan, pi
import pygame, time           # for keyboard multi-threaded inputs
from pygame.locals import *
import datetime

def getKeyboardCommands(stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):
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
    thr += dthr / 20.0

    return (stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

def getJoystickCommands(stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):
    pygame.event.get()
    drll = my_joystick.get_axis(3) # 0 = right stick lat
    dpch = my_joystick.get_axis(4) # 0 = right stick long
    dyaw = my_joystick.get_axis(0) # 0 = left stick lat
    dthr = my_joystick.get_axis(1) # 0 = left stick long
    # print(drll)

    # flush rest and display
    pygame.event.clear()

    #         if event.key == pygame.K_DELETE:
    #             stat = False # break drone
    #         if event.key == pygame.K_SPACE:
    #             rset = True # reset position/controls
    #             rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
    #             drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
    rll += drll / 100.0
    pch += dpch / 100.0
    yaw += dyaw / 100.0
    thr += dthr / 20.0

    return (stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)


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

pos = client.getPosition()
orq = client.getOrientation()
pos0,orq0 = (pos,orq)
pos_new,orq_new = (pos,orq)
ore0 = client.toEulerianAngle(orq)

rll_prv,pch_prv,yaw_prv,thr_prv = (0.0,0.0,0.0,0.0)
drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
drll_prv,dpch_prv,dyaw_prv,dthr_prv = (0.0,0.0,0.0,0.0)

orq_new = [0.0,0.0,0.0,0.0]
ore_new = [0.0,0.0,0.0]
ore_check = [0.0,0.0,0.0]

ps = 0

g = 9.81 # gravity [m/s^2]
m = 0.6 # mass [kg]
Ix,Iy,Iz = (0.00115,0.00115,0.00598) # inertia tensor principle components [kg m^2] (based on Gremillion 2016 https://arc.aiaa.org/doi/abs/10.2514/1.J054408)
Lp,Mq,Nr,Xu,Yv,Zw = (-0.01,-0.01,-0.2,-0.02,-0.02,-0.05) # aerodynamic drag derivatives [kg / s]

K_ph,K_p = (1.5,-0.02)
K_th,K_q = (K_ph,K_p)
K_r,K_dr = (1.0,0.0)
K_z,K_dz,K_z_i = (-20.0,5.0,-0.05)

# initialize states
ph,th,ps,p,q,r,u,v,w,x,y,z = (ore0[0],ore0[1],ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,pos0[0],pos0[1],pos0[2])
# initialize state derivatives
ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
# initialize state derivatives
dph,dth,dps,dp,dq,dr,du,dv,dw,dx,dy,dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
# initialize aerodynamic states
tawx,tawy,tawz,fwx,fwy,fwz = (0.0,0.0,0.0,0.0,0.0,0.0) # wind forces [N] torques [N m]

z_i = 0.0 # z error integral

dt = 0.05
while stat:
    rset = False

    # retrieve states from buffer
    ph,th,ps,p,q,r,u,v,w,x,y,z = (ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv)

    # retrieve commands from buffer
    rll,pch,yaw,thr = (rll_prv,pch_prv,yaw_prv,thr_prv)
    drll,dpch,dyaw,dthr = (drll_prv,dpch_prv,dyaw_prv,dthr_prv)

    # get commands from keyboard
    if control == 'keyboard':
        stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr = getKeyboardCommands(stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)
    elif control == 'joystick':
        stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr = getJoystickCommands(stat,rset,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

    # state reference
    ph_ref = rll
    th_ref = pch
    r_ref = yaw
    z_ref = thr + pos0[2]

    # state error
    ph_err = ph_ref - ph
    th_err = th_ref - th
    r_err = r_ref
    z_err = z_ref - z

    # z error integral
    z_i += z_err * dt

    # force/torque control
    tax = K_ph * ph_err + K_p * p
    tay = K_th * th_err + K_q * q
    taz = K_r * r_err
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
    # get vehicle status
    pos = client.getPosition() # get current position
    orq = client.getOrientation() # get current orientation
    collision = client.getCollisionInfo() # get collision status

    if collision[0] == True: # if collision, reset position/orientation/controls
        print("COLLISION - Resetting")
        client.simSetPose(pos0,orq0) # reset to origin
        pos_new,orq_new = (pos0,orq0) # update current position/orientation to origin
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        ph,th,ps,p,q,r,u,v,w,x,y,z = (ore0[0],ore0[1],ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,pos0[0],pos0[1],pos0[2])
        ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
        z_i = 0.0
    else: # if no collision update position/orientation
        # pos_new = [pos_new[0] + (th / -2.0)*cos(ps) - (ph / 2.0)*sin(ps), pos_new[1] + (ph / 2.0)*cos(ps) + (th / -2.0)*sin(ps), pos_new[2] + (thr / 2.0)]
        # orq_new = orq_new
        client.simSetPose(pos_new,orq_new)

    # send states to buffer
    ph_prv,th_prv,ps_prv,p_prv,q_prv,r_prv,u_prv,v_prv,w_prv,x_prv,y_prv,z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)

    # send commands to buffer
    rll_prv,pch_prv,yaw_prv,thr_prv = (rll,pch,yaw,thr)
    drll_prv,dpch_prv,dyaw_prv,dthr_prv = (drll,dpch,dyaw,dthr)

    # print debug output command/orientation
    print("yaw %f,ps %f,dps %f,r %f,dr %f,re %f,rr %f,t %f,oe %f,oq %f" %(yaw,ps,dps,r,dr,r_err,r_ref,taz,ore_new[2],ore_check[2]))
