import pygame
import time
import cv2
import numpy as np
from pygame.locals import *
import sys
import os

pygame.init()

camera = cv2.VideoCapture('Doctors.mp4')
camera2 = cv2.VideoCapture('Backg.mp4')
frame_counter = 0

display_width = 1920
display_height = 1080

condition_for_reset = 0

with open('screen1code.txt', 'r') as myfile:
    screen1_pass = myfile.read().replace('\n', '')
screen1_pass = screen1_pass.split(" ")

print ("Code SRN 1: ",screen1_pass)

with open('screen3code.txt', 'r') as myfile2:
    screen3_pass = myfile2.read().replace('\n', '')
screen3_pass = screen3_pass.split(" ")

print ("Code SRN 3: ",screen3_pass)

screen3_pass_var = []
cond_for_reset = 0

#### SCREEN 1 ####
s1_Q_x , s1_Q_y = display_width * 0.041667 , display_height * 0.625
s1_b_w , s1_b_h = display_width * 0.0591667 , display_height * 0.094444
s1_b_gapx , s1_b_gapy = display_width * 0.02 , display_height * 0.0319444

s1_W_x , s1_W_y = s1_Q_x + s1_b_w + s1_b_gapx , s1_Q_y
s1_E_x , s1_E_y = (s1_W_x + s1_b_w + s1_b_gapx)*0.99 , s1_W_y
s1_R_x , s1_R_y = (s1_E_x + s1_b_w + s1_b_gapx)*0.999 , s1_E_y
s1_T_x , s1_T_y = (s1_R_x + s1_b_w + s1_b_gapx)*0.994 , s1_R_y
s1_Y_x , s1_Y_y = (s1_T_x + s1_b_w + s1_b_gapx)*0.998 , s1_T_y
s1_U_x , s1_U_y = (s1_Y_x + s1_b_w + s1_b_gapx)*0.998 , s1_Y_y
s1_I_x , s1_I_y = (s1_U_x + s1_b_w + s1_b_gapx)*0.998 , s1_U_y
s1_O_x , s1_O_y = (s1_I_x + s1_b_w + s1_b_gapx)*0.997 , s1_I_y
s1_P_x , s1_P_y = (s1_O_x + s1_b_w + s1_b_gapx)*0.995 , s1_O_y

s1_A_x , s1_A_y = display_width * 0.065 , display_height * 0.75
s1_S_x , s1_S_y = (s1_A_x + s1_b_w + s1_b_gapx)*0.998 , s1_A_y
s1_D_x , s1_D_y = (s1_S_x + s1_b_w + s1_b_gapx)*0.99 , s1_S_y
s1_F_x , s1_F_y = s1_D_x + s1_b_w + s1_b_gapx , s1_D_y
s1_G_x , s1_G_y = (s1_F_x + s1_b_w + s1_b_gapx)*0.99 , s1_F_y
s1_H_x , s1_H_y = s1_G_x + s1_b_w + s1_b_gapx , s1_G_y
s1_J_x , s1_J_y = (s1_H_x + s1_b_w + s1_b_gapx)*0.997 , s1_H_y
s1_K_x , s1_K_y = s1_J_x + s1_b_w + s1_b_gapx , s1_J_y
s1_L_x , s1_L_y = (s1_K_x + s1_b_w + s1_b_gapx)*0.995 , s1_K_y

s1_Z_x , s1_Z_y = display_width * 0.141667 , (display_height * 0.879167)*0.998
s1_X_x , s1_X_y = s1_Z_x + s1_b_w + s1_b_gapx , s1_Z_y
s1_C_x , s1_C_y = (s1_X_x + s1_b_w + s1_b_gapx)*0.998 , s1_X_y
s1_V_x , s1_V_y = (s1_C_x + s1_b_w + s1_b_gapx)*0.994 , s1_C_y
s1_B_x , s1_B_y = s1_V_x + s1_b_w + s1_b_gapx , s1_V_y
s1_N_x , s1_N_y = (s1_B_x + s1_b_w + s1_b_gapx)*0.995 , s1_B_y
s1_M_x , s1_M_y = (s1_N_x + s1_b_w + s1_b_gapx)*0.998 , s1_N_y

s1_backspace_x , s1_backspace_y = display_width * 0.8333 , display_height * 0.625
s1_backspace_w , s1_backspace_h = display_width * 0.1341667 , display_height * 0.094444

s1_submit_x , s1_submit_y = display_width * 0.831667 , display_height * 0.748611
s1_submit_w , s1_submit_h = display_width * 0.1341667 , display_height * 0.216667

#### SCREEN 3 ####

block_width_x = display_width * 0.08583333
block_height_y = display_height * 0.1527778
block_gap_x = display_width * 0.01
block_gap_y = display_height * 0.01389

fit_X, fit_Y = 0, 0

doc_1_X , doc_1_Y = display_width*0.07, display_height*0.767
doc_2_X , doc_2_Y = display_width*0.07+block_width_x+block_gap_x, display_height*0.767
doc_3_X , doc_3_Y = display_width*0.07+block_width_x*2+block_gap_x*2,display_height*0.767
doc_4_X , doc_4_Y = display_width*0.07+block_width_x*3+block_gap_x*3,display_height*0.767
doc_5_X , doc_5_Y = display_width*0.07+block_width_x*4+block_gap_x*4,display_height*0.767

Exp_im_X , Exp_im_Y = display_width*0.65166, display_height*0.2625
pat_1_X , pat_1_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625
pat_2_X , pat_2_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625
pat_3_X , pat_3_Y = display_width*0.65166, display_height*0.2625+block_height_y+block_gap_y
pat_4_X , pat_4_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y+block_gap_y
pat_5_X , pat_5_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y+block_gap_y
pat_6_X , pat_6_Y = display_width*0.65166,display_height*0.2625+block_height_y*2+block_gap_y*2
pat_7_X , pat_7_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y*2+block_gap_y*2
pat_8_X , pat_8_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y*2+block_gap_y*2
pat_9_X , pat_9_Y = display_width*0.65166, display_height*0.2625+block_height_y*3+block_gap_y*3
pat_10_X , pat_10_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y*3+block_gap_y*3
pat_11_X , pat_11_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y*3+block_gap_y*3

#SCREEN3 RESET BUTTON
res_condition = False
res_dialog_x , res_dialog_y = display_width * 0.183333, display_height * 0.3472222
res_X , res_Y = display_width * 0.7975 , display_height * 0.119444
res_X_w , res_Y_h = display_width * 0.059167 , display_height * 0.1069444
b_res_lx , b_res_ly = display_width * 0.24167 , display_height * 0.486111
b_res_hx , b_res_hy = display_width * 0.4875 , display_height * 0.5763889
b_cancel_lx , b_cancel_ly = display_width * 0.508333 , display_height * 0.119444
b_cancel_hx , b_cancel_hy = display_width * 0.755833 , display_height * 0.5763889

#SCREEN3 ADMIN BUTTON
admin_X , admin_Y = display_width * 0.865 , display_height * 0.9225
admin_dialog_x , admin_dialog_y = display_width * 0.0958333 , display_height * 0.35
b_res2_lx , b_res2_ly = display_width * 0.5358333 , display_height * 0.4375
b_res2_hx , b_res2_hy = display_width * 0.68167 , display_height * 0.5
b_exit_lx , b_exit_ly = display_width * 0.6958333 , display_height * 0.4375
b_exit_hx , b_exit_hy = display_width * 0.8433333 , display_height * 0.5
b_button_lx , b_button_ly = display_width * 0.1108333 , display_height * 0.4402778
b_button_w , b_button_h = display_width * 0.0375 , display_height * 0.0625
b_button_gapx , b_button_gapy = display_width * 0.0116667 , display_height * 0.0194444

pat1_drag = False
pat2_drag = False
pat3_drag = False
pat4_drag = False
pat5_drag = False
pat6_drag = False
pat7_drag = False
pat8_drag = False
pat9_drag = False
pat10_drag = False
pat11_drag = False

prev_1x, prev_1y = 0, 0
prev_2x, prev_2y = 0, 0
prev_3x, prev_3y = 0, 0
prev_4x, prev_4y = 0, 0
prev_5x, prev_5y = 0, 0
prev_6x, prev_6y = 0, 0
prev_7x, prev_7y = 0, 0
prev_8x, prev_8y = 0, 0
prev_9x, prev_9y = 0, 0
prev_10x, prev_10y = 0, 0
prev_11x, prev_11y = 0, 0

open_zoomX , open_zoomY = display_width * 0.191667 , display_height * 0.0625

#rectangle = pygame.rect.Rect(pat_1_X, pat_1_Y, block_width_x, block_height_y)
#rectangle_draging = False

bgx , bgy = display_width * 0.0108333, display_height * 0.01805556
b1x , b1y , b1w , b1h = display_width * 0.0708333 , display_height * 0.09027778 , display_width * 0.085 , display_height * 0.15
b2x , b2y = b1x , b1y + b1h + bgy
b3x , b3y = b1x , b2y + b1h + bgy
b4x , b4y = b1x , b3y + b1h + bgy

b5x , b5y = b1x + b1w + bgx , b1y
b6x , b6y = b1x + b1w + bgx , b1y + b1h + bgy
b7x , b7y = b1x + b1w + bgx , b2y + b1h + bgy
b8x , b8y = b1x + b1w + bgx , b3y + b1h + bgy

b9x , b9y = b5x + b1w + bgx , b1y
b10x , b10y = b5x + b1w + bgx , b1y + b1h + bgy
b11x , b11y = b5x + b1w + bgx , b2y + b1h + bgy
b12x , b12y = b5x + b1w + bgx , b3y + b1h + bgy

b13x , b13y = b9x + b1w + bgx , b1y
b14x , b14y = b9x + b1w + bgx , b1y + b1h + bgy
b15x , b15y = b9x + b1w + bgx , b2y + b1h + bgy
b16x , b16y = b9x + b1w + bgx , b3y + b1h + bgy

b17x , b17y = b13x + b1w + bgx , b1y
b18x , b18y = b13x + b1w + bgx , b1y + b1h + bgy
b19x , b19y = b13x + b1w + bgx , b2y + b1h + bgy
b20x , b20y = b13x + b1w + bgx , b3y + b1h + bgy

pat1_ = False
pat2_ = False
pat3_ = False
pat4_ = False
pat5_ = False
pat6_ = False
pat7_ = False
pat8_ = False
pat9_ = False
pat10_ = False
pat11_ = False
priority = "NONE"

r1c1,r1c2,r1c3,r1c4,r1c5 = 0,0,0,0,0
r2c1,r2c2,r2c3,r2c4,r2c5 = 0,0,0,0,0
r3c1,r3c2,r3c3,r3c4,r3c5 = 0,0,0,0,0
r4c1,r4c2,r4c3,r4c4,r4c5 = 0,0,0,0,0

c1,c2,c3,c4,c5 = 0,0,0,0,0

background = pygame.image.load('background.png')
background = pygame.transform.scale(background,(display_width,display_height))
#gameDisplay.blit(background,(0,0))
#pygame.display.flip()
Layout_im = pygame.image.load('Layout.png')
Layout_im = pygame.transform.scale(Layout_im,(display_width,display_height))
#gameDisplay.blit(Layout_im,(0,0))

doc_1 = pygame.image.load('Doctor_1.jpg')
doc_1 = pygame.transform.scale(doc_1,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(doc_1,(doc_1_X, doc_1_Y))

doc_2 = pygame.image.load('Doctor_2.jpg')
doc_2 = pygame.transform.scale(doc_2,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(doc_2,(doc_2_X, doc_2_Y))

doc_3 = pygame.image.load('Doctor_3.jpg')
doc_3 = pygame.transform.scale(doc_3,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(doc_3,(doc_3_X, doc_3_Y))

doc_4 = pygame.image.load('Doctor_4.jpg')
doc_4 = pygame.transform.scale(doc_4,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(doc_4,(doc_4_X, doc_4_Y))

doc_5 = pygame.image.load('Doctor_5.jpg')
doc_5 = pygame.transform.scale(doc_5,(int(block_width_x),int(block_height_y)))

Exp_im = pygame.image.load('Explainer.jpg')
Exp_im = pygame.transform.scale(Exp_im,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(Exp_im,(Exp_im_X, Exp_im_Y))

res_msg = pygame.image.load('Reset_game.png')
res_msg = pygame.transform.scale(res_msg,(int(display_width*0.6375),int(display_height*0.2527778)))

admin_msg = pygame.image.load('Admin_game.png')
admin_msg = pygame.transform.scale(admin_msg, (int(display_width * 0.7766667),int(display_height*0.2638889)))


pat_1 = pygame.image.load('Patient_1.jpg')
pat_1 = pygame.transform.scale(pat_1,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_1,(pat_1_X, pat_1_Y))

pat_2 = pygame.image.load('Patient_2.jpg')
pat_2 = pygame.transform.scale(pat_2,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_2,(pat_2_X, pat_2_Y))

pat_3 = pygame.image.load('Patient_3.jpg')
pat_3 = pygame.transform.scale(pat_3,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_3,(pat_3_X, pat_3_Y))

pat_4 = pygame.image.load('Patient_4.jpg')
pat_4 = pygame.transform.scale(pat_4,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_4,(pat_4_X, pat_3_Y))

pat_5 = pygame.image.load('Patient_5.jpg')
pat_5 = pygame.transform.scale(pat_5,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_5,(pat_5_X, pat_5_Y))

pat_6 = pygame.image.load('Patient_6.jpg')
pat_6 = pygame.transform.scale(pat_6,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_6,(pat_6_X, pat_6_Y))

pat_7 = pygame.image.load('Patient_7.jpg')
pat_7 = pygame.transform.scale(pat_7,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_7,(pat_7_X, pat_7_Y))

pat_8 = pygame.image.load('Patient_8.jpg')
pat_8 = pygame.transform.scale(pat_8,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_8,(pat_8_X, pat_8_Y))

pat_9 = pygame.image.load('Patient_9.jpg')
pat_9 = pygame.transform.scale(pat_9,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_9,(pat_9_X, pat_9_Y))

pat_10 = pygame.image.load('Patient_10.jpg')
pat_10 = pygame.transform.scale(pat_10,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_10,(pat_10_X, pat_10_Y))

pat_11 = pygame.image.load('Patient_11.jpg')
pat_11 = pygame.transform.scale(pat_11,(int(block_width_x),int(block_height_y)))
#gameDisplay.blit(pat_11,(pat_11_X, pat_11_Y))

exp_hd = pygame.image.load('Ex_D.jpg')
exp_hd = pygame.transform.scale(exp_hd,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_1 = pygame.image.load('PD_1.jpg')
pat_hd_1 = pygame.transform.scale(pat_hd_1,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_2 = pygame.image.load('PD_2.jpg')
pat_hd_2 = pygame.transform.scale(pat_hd_2,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_3 = pygame.image.load('PD_3.jpg')
pat_hd_3 = pygame.transform.scale(pat_hd_3,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_4 = pygame.image.load('PD_4.jpg')
pat_hd_4 = pygame.transform.scale(pat_hd_4,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_5 = pygame.image.load('PD_5.jpg')
pat_hd_5 = pygame.transform.scale(pat_hd_5,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_6 = pygame.image.load('PD_6.jpg')
pat_hd_6 = pygame.transform.scale(pat_hd_6,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_7 = pygame.image.load('PD_7.jpg')
pat_hd_7 = pygame.transform.scale(pat_hd_7,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_8 = pygame.image.load('PD_8.jpg')
pat_hd_8 = pygame.transform.scale(pat_hd_8,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_9 = pygame.image.load('PD_9.jpg')
pat_hd_9 = pygame.transform.scale(pat_hd_9,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_10 = pygame.image.load('PD_10.jpg')
pat_hd_10 = pygame.transform.scale(pat_hd_10,(int(display_width*0.6),int(display_height*0.88)))
pat_hd_11 = pygame.image.load('PD_11.jpg')
pat_hd_11 = pygame.transform.scale(pat_hd_11,(int(display_width*0.6),int(display_height*0.88)))
doc_hd_1 = pygame.image.load('DD_1.jpg')
doc_hd_1 = pygame.transform.scale(doc_hd_1,(int(display_width*0.6),int(display_height*0.88)))
doc_hd_2 = pygame.image.load('DD_2.jpg')
doc_hd_2 = pygame.transform.scale(doc_hd_2,(int(display_width*0.6),int(display_height*0.88)))
doc_hd_3 = pygame.image.load('DD_3.jpg')
doc_hd_3 = pygame.transform.scale(doc_hd_3,(int(display_width*0.6),int(display_height*0.88)))
doc_hd_4 = pygame.image.load('DD_4.jpg')
doc_hd_4 = pygame.transform.scale(doc_hd_4,(int(display_width*0.6),int(display_height*0.88)))
doc_hd_5 = pygame.image.load('DD_5.jpg')
doc_hd_5 = pygame.transform.scale(doc_hd_5,(int(display_width*0.6),int(display_height*0.88)))



####

screen1_pass_var = []
screen1_pass_count = 0

black = (0,0,0)
red = (255,100,100)
pure_red = (255,0,0)
white = (255,255,255)


gameDisplay = pygame.display.set_mode((display_width,display_height),pygame.FULLSCREEN)
pygame.display.set_caption("Testing")
clock = pygame.time.Clock()
screen1 = True

def text_objects(text, font, colour):
    textSurface = font.render(text, True, colour)
    return textSurface, textSurface.get_rect()

def text_display(colour):
    largeText = pygame.font.SysFont("cairo",int(display_width * 0.09583333))
    print_code = ''.join(screen1_pass_var)
    TextSurf, TextRect = text_objects(print_code, largeText, colour)
    TextRect.center = ((display_width/2),(display_height/2.22))
    gameDisplay.blit(TextSurf, TextRect)


def text_objects2(text, font, colour):
    textSurface = font.render(text, True, colour)
    return textSurface, textSurface.get_rect()

def text_display2(colour):
    largeText = pygame.font.SysFont("cairo",int(display_width * 0.0333333))
    print_code = ''.join(screen3_pass_var)
    TextSurf, TextRect = text_objects(print_code, largeText, colour)
    TextRect.center = ((display_width * 0.443333),(display_height * 0.472222))
    gameDisplay.blit(TextSurf, TextRect)

def text_objects3(text, font, colour):
    textSurface = font.render(text, True, colour)
    return textSurface, textSurface.get_rect()

def text_display3(colour,width_txt,height_txt,c_1):
    largeText = pygame.font.SysFont("cairo",int(display_width * 0.06))
    TextSurf, TextRect = text_objects(str(c_1), largeText, colour)
    TextRect.center = ((width_txt),(height_txt))
    gameDisplay.blit(TextSurf, TextRect)
    

def Video3():
    ret, frame = camera2.read()
        
    gameDisplay.fill([0,0,0])
    global frame_counter
    frame_counter += 1
    if frame_counter == camera2.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        camera2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if ret is True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        pass
    frame = np.rot90(frame)
    frame = cv2.flip( frame, 0 )
    
    frame = pygame.surfarray.make_surface(frame)
    gameDisplay.blit(frame, (0,0))


def screen3():
    screen3 = True
    global pat1_drag
    global pat2_drag
    global pat3_drag
    global pat4_drag
    global pat5_drag
    global pat6_drag
    global pat7_drag
    global pat8_drag
    global pat9_drag
    global pat10_drag
    global pat11_drag
    global prev_1x, prev_1y
    global prev_2x, prev_2y
    global prev_3x, prev_3y
    global prev_4x, prev_4y
    global prev_5x, prev_5y
    global prev_6x, prev_6y
    global prev_7x, prev_7y
    global prev_8x, prev_8y
    global prev_9x, prev_9y
    global prev_10x, prev_10y
    global prev_11x, prev_11y
    global pat1_
    global pat2_
    global pat3_
    global pat4_
    global pat5_
    global pat6_
    global pat7_
    global pat8_
    global pat9_
    global pat10_
    global pat11_
    global fit_X , fit_Y
    global c1,c2,c3,c4,c5,r1c1,r2c1
    global priority
    global pat_1_X,pat_1_Y
    global pat_2_X,pat_2_Y
    global pat_3_X,pat_3_Y
    global pat_4_X,pat_4_Y
    global pat_5_X,pat_5_Y
    global pat_6_X,pat_6_Y
    global pat_7_X,pat_7_Y
    global pat_8_X,pat_8_Y
    global pat_9_X,pat_9_Y
    global pat_10_X,pat_10_Y
    global pat_11_X,pat_11_Y
    
    global r1c1,r1c2,r1c3,r1c4,r1c5
    global r2c1,r2c2,r2c3,r2c4,r2c5
    global r3c1,r3c2,r3c3,r3c4,r3c5
    global r4c1,r4c2,r4c3,r4c4,r4c5
    global condition_for_reset
    while screen3:
        
            #print mouse
            for event in pygame.event.get():
                #print event
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse = pygame.mouse.get_pos()
                        #pygame.draw.rect(gameDisplay,red,(pat_1_X + block_width_x,pat_1_Y + (block_height_y / 3.4),block_width_x,block_height_y))
                        #pygame.display.update()
                        condition_for_reset = 0
                        if pat_1_X < mouse[0] < pat_1_X + block_width_x  and pat_1_Y < mouse[1] < pat_1_Y + block_height_y:
                            pat1_drag = True
                            pat1_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_1_X - mouse_x
                            offset_y = pat_1_Y - mouse_y
                            prev_1x , prev_1y = pat_1_X , pat_1_Y
                            priority = "P1"
                        elif pat_2_X < mouse[0] < pat_2_X + block_width_x  and pat_2_Y < mouse[1] < pat_2_Y + block_height_y:
                            pat2_drag = True
                            pat2_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_2_X - mouse_x
                            offset_y = pat_2_Y - mouse_y
                            prev_2x , prev_2y = pat_2_X , pat_2_Y
                            priority = "P2"
                        elif pat_3_X  < mouse[0] < pat_3_X + block_width_x  and pat_3_Y < mouse[1] < pat_3_Y + block_height_y:
                            pat3_drag = True
                            pat3_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_3_X - mouse_x
                            offset_y = pat_3_Y - mouse_y
                            prev_3x , prev_3y = pat_3_X , pat_3_Y
                            priority = "P3"
                        elif pat_4_X < mouse[0] < pat_4_X + block_width_x and pat_4_Y < mouse[1] < pat_4_Y + block_height_y:
                            pat4_drag = True
                            pat4_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_4_X - mouse_x
                            offset_y = pat_4_Y - mouse_y
                            prev_4x , prev_4y = pat_4_X , pat_4_Y
                            priority = "P4"
                        elif pat_5_X < mouse[0] < pat_5_X + block_width_x and pat_5_Y < mouse[1] < pat_5_Y + block_height_y:
                            pat5_drag = True
                            pat5_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_5_X - mouse_x
                            offset_y = pat_5_Y - mouse_y
                            prev_5x , prev_5y = pat_5_X , pat_5_Y
                            priority = "P5"
                        elif pat_6_X < mouse[0] < pat_6_X + block_width_x  and pat_6_Y < mouse[1] < pat_6_Y + block_height_y:
                            pat6_drag = True
                            pat6_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_6_X - mouse_x
                            offset_y = pat_6_Y - mouse_y
                            prev_6x , prev_6y = pat_6_X , pat_6_Y
                            priority = "P6"
                        elif pat_7_X < mouse[0] < pat_7_X + block_width_x and pat_7_Y < mouse[1] < pat_7_Y + block_height_y:
                            pat7_drag = True
                            pat7_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_7_X - mouse_x
                            offset_y = pat_7_Y - mouse_y
                            prev_7x , prev_7y = pat_7_X , pat_7_Y
                            priority = "P7"
                        elif pat_8_X < mouse[0] < pat_8_X + block_width_x  and pat_8_Y < mouse[1] < pat_8_Y + block_height_y:
                            pat8_drag = True
                            pat8_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_8_X - mouse_x
                            offset_y = pat_8_Y - mouse_y
                            prev_8x , prev_8y = pat_8_X , pat_8_Y
                            priority = "P8"
                        elif pat_9_X < mouse[0] < pat_9_X + block_width_x  and pat_9_Y < mouse[1] < pat_9_Y + block_height_y:
                            pat9_drag = True
                            pat9_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_9_X - mouse_x
                            offset_y = pat_9_Y - mouse_y
                            prev_9x , prev_9y = pat_9_X , pat_9_Y
                            priority = "P9"
                        elif pat_10_X < mouse[0] < pat_10_X + block_width_x  and pat_10_Y < mouse[1] < pat_10_Y + block_height_y:
                            pat10_drag = True
                            pat10_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_10_X - mouse_x
                            offset_y = pat_10_Y - mouse_y
                            prev_10x , prev_10y = pat_10_X , pat_10_Y
                            priority = "P10"
                        elif pat_11_X < mouse[0] < pat_11_X + block_width_x  and pat_11_Y < mouse[1] < pat_11_Y + block_height_y:
                            pat11_drag = True
                            pat11_ = True
                            mouse_x, mouse_y = event.pos
                            offset_x = pat_11_X - mouse_x
                            offset_y = pat_11_Y - mouse_y
                            prev_11x , prev_11y = pat_11_X , pat_11_Y
                            priority = "P11"
                        elif res_X < mouse[0] < res_X + res_X_w and res_Y < mouse[1] < res_Y + res_Y_h:
                            res_condition = True
                            while res_condition:
                                #print "RESET"
                                
                                #click = pygame.mouse.get_pressed()
                                #print mouse, click
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        pygame.quit()
                                        quit()
                                    if event.type == pygame.MOUSEBUTTONDOWN:
                                        if event.button == 1:
                                            mouse = pygame.mouse.get_pos()
                                            if b_res_lx < mouse[0] < b_res_hx and b_res_ly < mouse[1] < b_res_hy:
                                                print("on reset button")
                                                pat_1_X , pat_1_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625
                                                pat_2_X , pat_2_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625
                                                pat_3_X , pat_3_Y = display_width*0.65166, display_height*0.2625+block_height_y+block_gap_y
                                                pat_4_X , pat_4_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y+block_gap_y
                                                pat_5_X , pat_5_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y+block_gap_y
                                                pat_6_X , pat_6_Y = display_width*0.65166,display_height*0.2625+block_height_y*2+block_gap_y*2
                                                pat_7_X , pat_7_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y*2+block_gap_y*2
                                                pat_8_X , pat_8_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y*2+block_gap_y*2
                                                pat_9_X , pat_9_Y = display_width*0.65166, display_height*0.2625+block_height_y*3+block_gap_y*3
                                                pat_10_X , pat_10_Y = display_width*0.65166+block_width_x+block_gap_x, display_height*0.2625+block_height_y*3+block_gap_y*3
                                                pat_11_X , pat_11_Y = display_width*0.65166+block_width_x*2+block_gap_x*2, display_height*0.2625+block_height_y*3+block_gap_y*3
                                                prev_1y = 0
                                                prev_2y = 0
                                                prev_3y = 0
                                                prev_4y = 0
                                                prev_5y = 0
                                                prev_6y = 0
                                                prev_7y = 0
                                                prev_8y = 0
                                                prev_9y = 0
                                                prev_10y = 0
                                                prev_11y = 0
                                                c1,r1c1,r2c1,r3c1,r4c1 = 0,0,0,0,0
                                                c2,r1c2,r2c2,r3c2,r4c2 = 0,0,0,0,0
                                                c3,r1c3,r2c3,r3c3,r4c3 = 0,0,0,0,0
                                                c4,r1c4,r2c4,r3c4,r4c4 = 0,0,0,0,0
                                                c5,r1c5,r2c5,r3c5,r4c5 = 0,0,0,0,0
                                                fit_X, fit_Y = 0, 0
                                                res_condition = False
                                                #condition_for_reset = 1
                                            elif b_cancel_lx < mouse[0] < b_cancel_hx and b_cancel_ly < mouse[1] < b_cancel_hy:
                                                print("on cancel button")
                                                res_condition = False
                                gameDisplay.blit(res_msg,(res_dialog_x, res_dialog_y))
                                pygame.display.flip()
                        elif admin_X < mouse[0] < admin_X + res_X_w and res_Y < mouse[1] < res_Y + res_Y_h:
                                admin_condition = True
                                global screen3_pass_var
                                global screen3_pass_count
                                del screen3_pass_var[:]
                                screen3_pass_count = 0
                                while admin_condition:
                                    #click = pygame.mouse.get_pressed()
                                    #print mouse
                                    for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONUP:
                                            click = event.button
                                            mouse = pygame.mouse.get_pos()
                                            if b_res2_lx < mouse[0] < b_res2_hx and b_res2_ly < mouse[1] < b_res2_hy and click == 1:
                                                print("on reset game")
                                                if screen3_pass == screen3_pass_var:
                                                    #os.system('python "H:\Pygame project\Files and Project\coding2.py"')
                                                    os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
                                                    #os.execv('H:\Pygame\Files\coding2.py', [''])
                                                else:
                                                    text_display2(red)
                                                    pygame.display.update()
                                                    time.sleep(1)
                                                    del screen3_pass_var[:]
                                                    admin_condition = False
                                            elif b_exit_lx < mouse[0] < b_exit_hx and b_exit_ly < mouse[1] < b_exit_hy and click == 1:
                                                if screen3_pass == screen3_pass_var:
                                                    #os.system("sudo poweroff")
                                                    pygame.quit()
                                                    quit()
                                                else:
                                                    text_display2(red)
                                                    pygame.display.update()
                                                    time.sleep(1)
                                                    print("on exit")
                                                    del screen3_pass_var[:]
                                                    admin_condition = False
                                            elif b_button_lx < mouse[0] < b_button_lx + b_button_w and b_button_ly < mouse[1] < b_button_ly + b_button_h and click== 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('1')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w + b_button_gapx < mouse[0] < b_button_lx + b_button_w*2 + b_button_gapx and b_button_ly < mouse[1] < b_button_ly+b_button_h and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('2')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*2 +b_button_gapx*2 < mouse[0] < b_button_lx + b_button_w*3 +b_button_gapx*2 and b_button_ly < mouse[1] < b_button_ly+b_button_h and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('3')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*3 +b_button_gapx*3 < mouse[0] < b_button_lx + b_button_w*4 +b_button_gapx*3 and b_button_ly < mouse[1] < b_button_ly+b_button_h and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('4')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*4 +b_button_gapx*4 < mouse[0] < b_button_lx + b_button_w*5 +b_button_gapx*4 and b_button_ly < mouse[1] < b_button_ly+b_button_h and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('5')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx < mouse[0] < b_button_lx + b_button_w and b_button_ly + b_button_h + b_button_gapy < mouse[1] < b_button_ly + b_button_h*2 + b_button_gapy and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('6')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w + b_button_gapx < mouse[0] < b_button_lx + b_button_w*2 + b_button_gapx and b_button_ly + b_button_h + b_button_gapy < mouse[1] < b_button_ly + b_button_h*2 + b_button_gapy and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('7')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*2 +b_button_gapx*2 < mouse[0] < b_button_lx + b_button_w*3 +b_button_gapx*2 and b_button_ly + b_button_h + b_button_gapy < mouse[1] < b_button_ly + b_button_h*2 + b_button_gapy and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('8')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*3 +b_button_gapx*3 < mouse[0] < b_button_lx + b_button_w*4 +b_button_gapx*3 and b_button_ly + b_button_h + b_button_gapy < mouse[1] < b_button_ly + b_button_h*2 + b_button_gapy and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('9')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                            elif b_button_lx + b_button_w*4 +b_button_gapx*4 < mouse[0] < b_button_lx + b_button_w*5 +b_button_gapx*4 and b_button_ly + b_button_h + b_button_gapy < mouse[1] < b_button_ly + b_button_h*2 + b_button_gapy and click == 1 and screen3_pass_count < 4:
                                                screen3_pass_var.append('0')
                                                screen3_pass_count += 1
                                                print (screen3_pass_var)
                                                print (screen3_pass == screen3_pass_var)
                                        
##                                      
                                    gameDisplay.blit(admin_msg,(admin_dialog_x, admin_dialog_y))
                                    text_display2(black)
                                    #time.sleep(0.2)
                                    pygame.display.update()
                                    
                            #elif admin_X

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        #print pygame.mouse.get_pos()
                        pat1_drag = False
                        pat2_drag = False
                        pat3_drag = False
                        pat4_drag = False
                        pat5_drag = False
                        pat6_drag = False
                        pat7_drag = False
                        pat8_drag = False
                        pat9_drag = False
                        pat10_drag = False
                        pat11_drag = False
                        opening = True
                        mouse = pygame.mouse.get_pos()
                        if pat1_ == True:
                            fit_X , fit_Y = pat_1_X + 30 , pat_1_Y + 25
                        elif pat2_ == True:
                            fit_X , fit_Y = pat_2_X + 30 , pat_2_Y + 25
                        elif pat3_ == True:
                            fit_X , fit_Y = pat_3_X + 30 , pat_3_Y + 25
                        elif pat4_ == True:
                            fit_X , fit_Y = pat_4_X + 30 , pat_4_Y + 25
                        elif pat5_ == True:
                            fit_X , fit_Y = pat_5_X + 30 , pat_5_Y + 25
                        elif pat6_ == True:
                            fit_X , fit_Y = pat_6_X + 30 , pat_6_Y + 25
                        elif pat7_ == True:
                            fit_X , fit_Y = pat_7_X + 30 , pat_7_Y + 25
                        elif pat8_ == True:
                            fit_X , fit_Y = pat_8_X + 30 , pat_8_Y + 25
                        elif pat9_ == True:
                            fit_X , fit_Y = pat_9_X + 30 , pat_9_Y + 25
                        elif pat10_ == True:
                            fit_X , fit_Y = pat_10_X + 30 , pat_10_Y + 25
                        elif pat11_ == True:
                            fit_X , fit_Y = pat_11_X + 30 , pat_11_Y+ 25
                        print (fit_X,fit_Y)
                        if b1x < fit_X < b1x+b1w and b1y < fit_Y < b1y+b1h and r1c1==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b1x , b1y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b1x , b1y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b1x , b1y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b1x , b1y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b1x , b1y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b1x , b1y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b1x , b1y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b1x , b1y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b1x , b1y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b1x , b1y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b2x , b2y
                        if b2x < fit_X < b2x+b1w and b2y < fit_Y < b2y+b1h and r2c1==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b2x , b2y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b2x , b2y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b2x , b2y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b2x , b2y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b2x , b2y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b2x , b2y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b2x , b2y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b2x , b2y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b2x , b2y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b2x , b2y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b2x , b2y
                        if b3x < fit_X < b3x+b1w and b3y < fit_Y < b3y+b1h and r3c1==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b3x , b3y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b3x , b3y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b3x , b3y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b3x , b3y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b3x , b3y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b3x , b3y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b3x , b3y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b3x , b3y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b3x , b3y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b3x , b3y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b3x , b3y
                        if b4x < fit_X < b4x+b1w and b4y < fit_Y < b4y+b1h and r4c1==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b4x , b4y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b4x , b4y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b4x , b4y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b4x , b4y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b4x , b4y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b4x , b4y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b4x , b4y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b4x , b4y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b4x , b4y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b4x , b4y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b4x , b4y
                        if b5x < fit_X < b5x+b1w and b5y < fit_Y < b5y+b1h and r1c2==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b5x , b5y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b5x , b5y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b5x , b5y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b5x , b5y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b5x , b5y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b5x , b5y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b5x , b5y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b5x , b5y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b5x , b5y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b5x , b5y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b5x , b5y
                        if b6x < fit_X < b6x+b1w and b6y < fit_Y < b6y+b1h  and r2c2==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b6x , b6y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b6x , b6y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b6x , b6y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b6x , b6y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b6x , b6y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b6x , b6y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b6x , b6y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b6x , b6y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b6x , b6y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b6x , b6y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b6x , b6y
                        if b7x < fit_X < b7x+b1w and b7y < fit_Y < b7y+b1h and r3c2==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b7x , b7y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b7x , b7y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b7x , b7y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b7x , b7y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b7x , b7y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b7x , b7y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b7x , b7y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b7x , b7y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b7x , b7y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b7x , b7y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b7x , b7y

                        if b8x < fit_X < b8x+b1w and b8y < fit_Y < b8y+b1h and r4c2==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b8x , b8y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b8x , b8y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b8x , b8y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b8x , b8y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b8x , b8y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b8x , b8y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b8x , b8y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b8x , b8y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b8x , b8y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b8x , b8y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b8x , b8y

                        if b9x < fit_X < b9x+b1w and b9y < fit_Y < b9y+b1h and r1c3==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b9x , b9y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b9x , b9y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b9x , b9y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b9x , b9y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b9x , b9y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b9x , b9y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b9x , b9y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b9x , b9y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b9x , b9y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b9x , b9y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b9x , b9y

                        if b10x < fit_X < b10x+b1w and b10y < fit_Y < b10y+b1h and r2c3==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b10x , b10y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b10x , b10y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b10x , b10y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b10x , b10y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b10x , b10y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b10x , b10y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b10x , b10y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b10x , b10y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b10x , b10y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b10x , b10y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b10x , b10y

                        if b11x < fit_X < b11x+b1w and b11y < fit_Y < b11y+b1h and r3c3==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b11x , b11y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b11x , b11y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b11x , b11y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b11x , b11y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b11x , b11y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b11x , b11y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b11x , b11y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b11x , b11y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b11x , b11y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b11x , b11y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b11x , b11y

                        if b12x < fit_X < b12x+b1w and b12y < fit_Y < b12y+b1h and r4c3==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b12x , b12y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b12x , b12y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b12x , b12y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b12x , b12y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b12x , b12y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b12x , b12y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b12x , b12y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b12x , b12y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b12x , b12y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b12x , b12y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b12x , b12y

                        if b13x < fit_X < b13x+b1w and b13y < fit_Y < b13y+b1h and r1c4==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b13x , b13y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b13x , b13y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b13x , b13y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b13x , b13y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b13x , b13y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b13x , b13y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b13x , b13y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b13x , b13y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b13x , b13y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b13x , b13y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b13x , b13y

                        if b14x < fit_X < b14x+b1w and b14y < fit_Y < b14y+b1h and r2c4==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b14x , b14y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b14x , b14y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b14x , b14y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b14x , b14y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b14x , b14y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b14x , b14y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b14x , b14y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b14x , b14y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b14x , b14y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b14x , b14y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b14x , b14y

                        if b15x < fit_X < b15x+b1w and b15y < fit_Y < b15y+b1h and r3c4==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b15x , b15y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b15x , b15y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b15x , b15y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b15x , b15y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b15x , b15y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b15x , b15y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b15x , b15y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b15x , b15y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b15x , b15y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b15x , b15y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b15x , b15y

                        if b16x < fit_X < b16x+b1w and b16y < fit_Y < b16y+b1h and r4c4==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b16x , b16y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b16x , b16y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b16x , b16y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b16x , b16y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b16x , b16y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b16x , b16y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b16x , b16y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b16x , b16y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b16x , b16y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b16x , b16y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b16x , b16y

                        if b17x < fit_X < b17x+b1w and b17y < fit_Y < b17y+b1h and r1c5==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b17x , b17y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b17x , b17y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b17x , b17y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b17x , b17y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b17x , b17y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b17x , b17y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b17x , b17y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b17x , b17y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b17x , b17y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b17x , b17y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b17x , b17y

                        if b18x < fit_X < b18x+b1w and b18y < fit_Y < b18y+b1h and r2c5==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b18x , b18y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b18x , b18y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b18x , b18y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b18x , b18y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b18x , b18y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b18x , b18y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b18x , b18y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b18x , b18y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b18x , b18y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b18x , b18y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b18x , b18y

                        if b19x < fit_X < b19x+b1w and b19y < fit_Y < b19y+b1h and r3c5==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b19x , b19y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b19x , b19y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b19x , b19y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b19x , b19y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b19x , b19y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b19x , b19y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b19x , b19y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b19x , b19y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b19x , b19y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b19x , b19y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b19x , b19y

                        if b20x < fit_X < b20x+b1w and b20y < fit_Y < b20y+b1h and r4c5==0:
                            if pat1_ == True:
                                pat_1_X , pat_1_Y = b20x , b20y
                            if pat2_ == True:
                                pat_2_X , pat_2_Y = b20x , b20y
                            if pat3_ == True:
                                pat_3_X , pat_3_Y = b20x , b20y
                            if pat4_ == True:
                                pat_4_X , pat_4_Y = b20x , b20y
                            if pat5_ == True:
                                pat_5_X , pat_5_Y = b20x , b20y
                            if pat6_ == True:
                                pat_6_X , pat_6_Y = b20x , b20y
                            if pat7_ == True:
                                pat_7_X , pat_7_Y = b20x , b20y
                            if pat8_ == True:
                                pat_8_X , pat_8_Y = b20x , b20y
                            if pat9_ == True:
                                pat_9_X , pat_9_Y = b20x , b20y
                            if pat10_ == True:
                                pat_10_X , pat_10_Y = b20x , b20y
                            if pat11_ == True:
                                pat_11_X , pat_11_Y = b20x , b20y
                        

                                
                        #if (pat_1_X,pat_1_Y or pat_2_X,pat_2_Y or pat_3_X,pat_3_Y or pat_4_X,pat_4_Y or pat_5_X,pat_5_Y or pat_6_X,pat_6_Y or pat_7_X,pat_7_Y or pat_8_X,pat_8_Y or pat_9_X,pat_9_Y or pat_10_X,pat_10_Y or pat_11_X,pat_11_Y) == b1x,b1y:
                        #print b1x,b1y,pat_1_X,pat_1_Y,pat_2_X,pat_2_Y
                        if ((b1x==pat_1_X) and (b1y==pat_1_Y)) or ((b1x==pat_2_X) and (b1y==pat_2_Y)) or ((b1x==pat_3_X) and (b1y==pat_3_Y)) or ((b1x==pat_4_X) and (b1y==pat_4_Y)) or ((b1x==pat_5_X) and (b1y==pat_5_Y)) or ((b1x==pat_6_X) and (b1y==pat_6_Y)) or ((b1x==pat_7_X) and (b1y==pat_7_Y)) or ((b1x==pat_8_X) and (b1y==pat_8_Y)) or ((b1x==pat_9_X) and (b1y==pat_9_Y)) or ((b1x==pat_10_X) and (b1y==pat_10_Y)) or ((b1x==pat_11_X) and (b1y==pat_11_Y)):
                            r1c1 = 1
                        else:
                            r1c1 = 0
                        if ((b2x==pat_1_X) and (b2y==pat_1_Y)) or ((b2x==pat_2_X) and (b2y==pat_2_Y)) or ((b2x==pat_3_X) and (b2y==pat_3_Y)) or ((b2x==pat_4_X) and (b2y==pat_4_Y)) or ((b2x==pat_5_X) and (b2y==pat_5_Y)) or ((b2x==pat_6_X) and (b2y==pat_6_Y)) or ((b2x==pat_7_X) and (b2y==pat_7_Y)) or ((b2x==pat_8_X) and (b2y==pat_8_Y)) or ((b2x==pat_9_X) and (b2y==pat_9_Y)) or ((b2x==pat_10_X) and (b2y==pat_10_Y)) or ((b2x==pat_11_X) and (b2y==pat_11_Y)):
                            r2c1 = 1
                        else:
                            r2c1 = 0
                        if ((b3x==pat_1_X) and (b3y==pat_1_Y)) or ((b3x==pat_2_X) and (b3y==pat_2_Y)) or ((b3x==pat_3_X) and (b3y==pat_3_Y)) or ((b3x==pat_4_X) and (b3y==pat_4_Y)) or ((b3x==pat_5_X) and (b3y==pat_5_Y)) or ((b3x==pat_6_X) and (b3y==pat_6_Y)) or ((b3x==pat_7_X) and (b3y==pat_7_Y)) or ((b3x==pat_8_X) and (b3y==pat_8_Y)) or ((b3x==pat_9_X) and (b3y==pat_9_Y)) or ((b3x==pat_10_X) and (b3y==pat_10_Y)) or ((b3x==pat_11_X) and (b3y==pat_11_Y)):
                            r3c1 = 1
                        else:
                            r3c1 = 0
                        if ((b4x==pat_1_X) and (b4y==pat_1_Y)) or ((b4x==pat_2_X) and (b4y==pat_2_Y)) or ((b4x==pat_3_X) and (b4y==pat_3_Y)) or ((b4x==pat_4_X) and (b4y==pat_4_Y)) or ((b4x==pat_5_X) and (b4y==pat_5_Y)) or ((b4x==pat_6_X) and (b4y==pat_6_Y)) or ((b4x==pat_7_X) and (b4y==pat_7_Y)) or ((b4x==pat_8_X) and (b4y==pat_8_Y)) or ((b4x==pat_9_X) and (b4y==pat_9_Y)) or ((b4x==pat_10_X) and (b4y==pat_10_Y)) or ((b4x==pat_11_X) and (b4y==pat_11_Y)):
                            r4c1 = 1
                        else:
                            r4c1 = 0
                        if ((b5x==pat_1_X) and (b5y==pat_1_Y)) or ((b5x==pat_2_X) and (b5y==pat_2_Y)) or ((b5x==pat_3_X) and (b5y==pat_3_Y)) or ((b5x==pat_4_X) and (b5y==pat_4_Y)) or ((b5x==pat_5_X) and (b5y==pat_5_Y)) or ((b5x==pat_6_X) and (b5y==pat_6_Y)) or ((b5x==pat_7_X) and (b5y==pat_7_Y)) or ((b5x==pat_8_X) and (b5y==pat_8_Y)) or ((b5x==pat_9_X) and (b5y==pat_9_Y)) or ((b5x==pat_10_X) and (b5y==pat_10_Y)) or ((b5x==pat_11_X) and (b5y==pat_11_Y)):
                            r1c2 = 1
                        else:
                            r1c2 = 0
                        if ((b6x==pat_1_X) and (b6y==pat_1_Y)) or ((b6x==pat_2_X) and (b6y==pat_2_Y)) or ((b6x==pat_3_X) and (b6y==pat_3_Y)) or ((b6x==pat_4_X) and (b6y==pat_4_Y)) or ((b6x==pat_5_X) and (b6y==pat_5_Y)) or ((b6x==pat_6_X) and (b6y==pat_6_Y)) or ((b6x==pat_7_X) and (b6y==pat_7_Y)) or ((b6x==pat_8_X) and (b6y==pat_8_Y)) or ((b6x==pat_9_X) and (b6y==pat_9_Y)) or ((b6x==pat_10_X) and (b6y==pat_10_Y)) or ((b6x==pat_11_X) and (b6y==pat_11_Y)):
                            r2c2 = 1
                        else:
                            r2c2 = 0
                        if ((b7x==pat_1_X) and (b7y==pat_1_Y)) or ((b7x==pat_2_X) and (b7y==pat_2_Y)) or ((b7x==pat_3_X) and (b7y==pat_3_Y)) or ((b7x==pat_4_X) and (b7y==pat_4_Y)) or ((b7x==pat_5_X) and (b7y==pat_5_Y)) or ((b7x==pat_6_X) and (b7y==pat_6_Y)) or ((b7x==pat_7_X) and (b7y==pat_7_Y)) or ((b7x==pat_8_X) and (b7y==pat_8_Y)) or ((b7x==pat_9_X) and (b7y==pat_9_Y)) or ((b7x==pat_10_X) and (b7y==pat_10_Y)) or ((b7x==pat_11_X) and (b7y==pat_11_Y)):
                            r3c2 = 1
                        else:
                            r3c2 = 0
                        if ((b8x==pat_1_X) and (b8y==pat_1_Y)) or ((b8x==pat_2_X) and (b8y==pat_2_Y)) or ((b8x==pat_3_X) and (b8y==pat_3_Y)) or ((b8x==pat_4_X) and (b8y==pat_4_Y)) or ((b8x==pat_5_X) and (b8y==pat_5_Y)) or ((b8x==pat_6_X) and (b8y==pat_6_Y)) or ((b8x==pat_7_X) and (b8y==pat_7_Y)) or ((b8x==pat_8_X) and (b8y==pat_8_Y)) or ((b8x==pat_9_X) and (b8y==pat_9_Y)) or ((b8x==pat_10_X) and (b8y==pat_10_Y)) or ((b8x==pat_11_X) and (b8y==pat_11_Y)):
                            r4c2 = 1
                        else:
                            r4c2 = 0
                        if ((b9x==pat_1_X) and (b9y==pat_1_Y)) or ((b9x==pat_2_X) and (b9y==pat_2_Y)) or ((b9x==pat_3_X) and (b9y==pat_3_Y)) or ((b9x==pat_4_X) and (b9y==pat_4_Y)) or ((b9x==pat_5_X) and (b9y==pat_5_Y)) or ((b9x==pat_6_X) and (b9y==pat_6_Y)) or ((b9x==pat_7_X) and (b9y==pat_7_Y)) or ((b9x==pat_8_X) and (b9y==pat_8_Y)) or ((b9x==pat_9_X) and (b9y==pat_9_Y)) or ((b9x==pat_10_X) and (b9y==pat_10_Y)) or ((b9x==pat_11_X) and (b9y==pat_11_Y)):
                            r1c3 = 1
                        else:
                            r1c3 = 0
                        if ((b10x==pat_1_X) and (b10y==pat_1_Y)) or ((b10x==pat_2_X) and (b10y==pat_2_Y)) or ((b10x==pat_3_X) and (b10y==pat_3_Y)) or ((b10x==pat_4_X) and (b10y==pat_4_Y)) or ((b10x==pat_5_X) and (b10y==pat_5_Y)) or ((b10x==pat_6_X) and (b10y==pat_6_Y)) or ((b10x==pat_7_X) and (b10y==pat_7_Y)) or ((b10x==pat_8_X) and (b10y==pat_8_Y)) or ((b10x==pat_9_X) and (b10y==pat_9_Y)) or ((b10x==pat_10_X) and (b10y==pat_10_Y)) or ((b10x==pat_11_X) and (b10y==pat_11_Y)):
                            r2c3 = 1
                        else:
                            r2c3 = 0
                        if ((b11x==pat_1_X) and (b11y==pat_1_Y)) or ((b11x==pat_2_X) and (b11y==pat_2_Y)) or ((b11x==pat_3_X) and (b11y==pat_3_Y)) or ((b11x==pat_4_X) and (b11y==pat_4_Y)) or ((b11x==pat_5_X) and (b11y==pat_5_Y)) or ((b11x==pat_6_X) and (b11y==pat_6_Y)) or ((b11x==pat_7_X) and (b11y==pat_7_Y)) or ((b11x==pat_8_X) and (b11y==pat_8_Y)) or ((b11x==pat_9_X) and (b11y==pat_9_Y)) or ((b11x==pat_10_X) and (b11y==pat_10_Y)) or ((b11x==pat_11_X) and (b11y==pat_11_Y)):
                            r3c3 = 1
                        else:
                            r3c3 = 0
                        if ((b12x==pat_1_X) and (b12y==pat_1_Y)) or ((b12x==pat_2_X) and (b12y==pat_2_Y)) or ((b12x==pat_3_X) and (b12y==pat_3_Y)) or ((b12x==pat_4_X) and (b12y==pat_4_Y)) or ((b12x==pat_5_X) and (b12y==pat_5_Y)) or ((b12x==pat_6_X) and (b12y==pat_6_Y)) or ((b12x==pat_7_X) and (b12y==pat_7_Y)) or ((b12x==pat_8_X) and (b12y==pat_8_Y)) or ((b12x==pat_9_X) and (b12y==pat_9_Y)) or ((b12x==pat_10_X) and (b12y==pat_10_Y)) or ((b12x==pat_11_X) and (b12y==pat_11_Y)):
                            r4c3 = 1
                        else:
                            r4c3 = 0
                        if ((b13x==pat_1_X) and (b13y==pat_1_Y)) or ((b13x==pat_2_X) and (b13y==pat_2_Y)) or ((b13x==pat_3_X) and (b13y==pat_3_Y)) or ((b13x==pat_4_X) and (b13y==pat_4_Y)) or ((b13x==pat_5_X) and (b13y==pat_5_Y)) or ((b13x==pat_6_X) and (b13y==pat_6_Y)) or ((b13x==pat_7_X) and (b13y==pat_7_Y)) or ((b13x==pat_8_X) and (b13y==pat_8_Y)) or ((b13x==pat_9_X) and (b13y==pat_9_Y)) or ((b13x==pat_10_X) and (b13y==pat_10_Y)) or ((b13x==pat_11_X) and (b13y==pat_11_Y)):
                            r1c4 = 1
                        else:
                            r1c4 = 0
                        if ((b14x==pat_1_X) and (b14y==pat_1_Y)) or ((b14x==pat_2_X) and (b14y==pat_2_Y)) or ((b14x==pat_3_X) and (b14y==pat_3_Y)) or ((b14x==pat_4_X) and (b14y==pat_4_Y)) or ((b14x==pat_5_X) and (b14y==pat_5_Y)) or ((b14x==pat_6_X) and (b14y==pat_6_Y)) or ((b14x==pat_7_X) and (b14y==pat_7_Y)) or ((b14x==pat_8_X) and (b14y==pat_8_Y)) or ((b14x==pat_9_X) and (b14y==pat_9_Y)) or ((b14x==pat_10_X) and (b14y==pat_10_Y)) or ((b14x==pat_11_X) and (b14y==pat_11_Y)):
                            r2c4 = 1
                        else:
                            r2c4 = 0
                        if ((b15x==pat_1_X) and (b15y==pat_1_Y)) or ((b15x==pat_2_X) and (b15y==pat_2_Y)) or ((b15x==pat_3_X) and (b15y==pat_3_Y)) or ((b15x==pat_4_X) and (b15y==pat_4_Y)) or ((b15x==pat_5_X) and (b15y==pat_5_Y)) or ((b15x==pat_6_X) and (b15y==pat_6_Y)) or ((b15x==pat_7_X) and (b15y==pat_7_Y)) or ((b15x==pat_8_X) and (b15y==pat_8_Y)) or ((b15x==pat_9_X) and (b15y==pat_9_Y)) or ((b15x==pat_10_X) and (b15y==pat_10_Y)) or ((b15x==pat_11_X) and (b15y==pat_11_Y)):
                            r3c4 = 1
                        else:
                            r3c4 = 0
                        if ((b16x==pat_1_X) and (b16y==pat_1_Y)) or ((b16x==pat_2_X) and (b16y==pat_2_Y)) or ((b16x==pat_3_X) and (b16y==pat_3_Y)) or ((b16x==pat_4_X) and (b16y==pat_4_Y)) or ((b16x==pat_5_X) and (b16y==pat_5_Y)) or ((b16x==pat_6_X) and (b16y==pat_6_Y)) or ((b16x==pat_7_X) and (b16y==pat_7_Y)) or ((b16x==pat_8_X) and (b16y==pat_8_Y)) or ((b16x==pat_9_X) and (b16y==pat_9_Y)) or ((b16x==pat_10_X) and (b16y==pat_10_Y)) or ((b16x==pat_11_X) and (b16y==pat_11_Y)):
                            r4c4 = 1
                        else:
                            r4c4 = 0
                        if ((b17x==pat_1_X) and (b17y==pat_1_Y)) or ((b17x==pat_2_X) and (b17y==pat_2_Y)) or ((b17x==pat_3_X) and (b17y==pat_3_Y)) or ((b17x==pat_4_X) and (b17y==pat_4_Y)) or ((b17x==pat_5_X) and (b17y==pat_5_Y)) or ((b17x==pat_6_X) and (b17y==pat_6_Y)) or ((b17x==pat_7_X) and (b17y==pat_7_Y)) or ((b17x==pat_8_X) and (b17y==pat_8_Y)) or ((b17x==pat_9_X) and (b17y==pat_9_Y)) or ((b17x==pat_10_X) and (b17y==pat_10_Y)) or ((b17x==pat_11_X) and (b17y==pat_11_Y)):
                            r1c5 = 1
                        else:
                            r1c5 = 0
                        if ((b18x==pat_1_X) and (b18y==pat_1_Y)) or ((b18x==pat_2_X) and (b18y==pat_2_Y)) or ((b18x==pat_3_X) and (b18y==pat_3_Y)) or ((b18x==pat_4_X) and (b18y==pat_4_Y)) or ((b18x==pat_5_X) and (b18y==pat_5_Y)) or ((b18x==pat_6_X) and (b18y==pat_6_Y)) or ((b18x==pat_7_X) and (b18y==pat_7_Y)) or ((b18x==pat_8_X) and (b18y==pat_8_Y)) or ((b18x==pat_9_X) and (b18y==pat_9_Y)) or ((b18x==pat_10_X) and (b18y==pat_10_Y)) or ((b18x==pat_11_X) and (b18y==pat_11_Y)):
                            r2c5 = 1
                        else:
                            r2c5 = 0
                        if ((b19x==pat_1_X) and (b19y==pat_1_Y)) or ((b19x==pat_2_X) and (b19y==pat_2_Y)) or ((b19x==pat_3_X) and (b19y==pat_3_Y)) or ((b19x==pat_4_X) and (b19y==pat_4_Y)) or ((b19x==pat_5_X) and (b19y==pat_5_Y)) or ((b19x==pat_6_X) and (b19y==pat_6_Y)) or ((b19x==pat_7_X) and (b19y==pat_7_Y)) or ((b19x==pat_8_X) and (b19y==pat_8_Y)) or ((b19x==pat_9_X) and (b19y==pat_9_Y)) or ((b19x==pat_10_X) and (b19y==pat_10_Y)) or ((b19x==pat_11_X) and (b19y==pat_11_Y)):
                            r3c5 = 1
                        else:
                            r3c5 = 0
                        if ((b20x==pat_1_X) and (b20y==pat_1_Y)) or ((b20x==pat_2_X) and (b20y==pat_2_Y)) or ((b20x==pat_3_X) and (b20y==pat_3_Y)) or ((b20x==pat_4_X) and (b20y==pat_4_Y)) or ((b20x==pat_5_X) and (b20y==pat_5_Y)) or ((b20x==pat_6_X) and (b20y==pat_6_Y)) or ((b20x==pat_7_X) and (b20y==pat_7_Y)) or ((b20x==pat_8_X) and (b20y==pat_8_Y)) or ((b20x==pat_9_X) and (b20y==pat_9_Y)) or ((b20x==pat_10_X) and (b20y==pat_10_Y)) or ((b20x==pat_11_X) and (b20y==pat_11_Y)):
                            r4c5 = 1
                        else:
                            r4c5 = 0

                        c1 = r1c1+r2c1+r3c1+r4c1
                        c2 = r1c2+r2c2+r3c2+r4c2
                        c3 = r1c3+r2c3+r3c3+r4c3
                        c4 = r1c4+r2c4+r3c4+r4c4
                        c5 = r1c5+r2c5+r3c5+r4c5

                        #print c1,c2,c3,c4,c5
                        print (condition_for_reset)
                        #if (pat_1_X or pat_1_X
                        if pat_1_X == prev_1x and pat_1_Y == prev_1y and condition_for_reset==0:
                            print ("Open P_1")
                            while opening:
                                gameDisplay.blit(pat_hd_1,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_1x, prev_1y = 0, 0
                        elif pat_2_X == prev_2x and pat_2_Y == prev_2y and condition_for_reset==0:
                            print ("Open P_2")
                            while opening:
                                gameDisplay.blit(pat_hd_2,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_2x, prev_2y = 0, 0
                        elif pat_3_X == prev_3x and pat_3_Y == prev_3y and condition_for_reset==0:
                            print ("Open P_3")
                            while opening:
                                gameDisplay.blit(pat_hd_3,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_3x, prev_3y = 0, 0
                        elif pat_4_X == prev_4x and pat_4_Y == prev_4y and condition_for_reset==0:
                            print ("Open P_4")
                            while opening:
                                gameDisplay.blit(pat_hd_4,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_4x, prev_4y = 0, 0
                        elif pat_5_X == prev_5x and pat_5_Y == prev_5y and condition_for_reset==0:
                            print ("Open P_5")
                            while opening:
                                gameDisplay.blit(pat_hd_5,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_5x, prev_5y = 0, 0
                        elif pat_6_X == prev_6x and pat_6_Y == prev_6y and condition_for_reset==0:
                            print ("Open P_6")
                            while opening:
                                gameDisplay.blit(pat_hd_6,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_6x, prev_6y = 0, 0
                        elif pat_7_X == prev_7x and pat_7_Y == prev_7y and condition_for_reset==0:
                            print ("Open P_7")
                            while opening:
                                gameDisplay.blit(pat_hd_7,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_7x, prev_7y = 0, 0
                        elif pat_8_X == prev_8x and pat_8_Y == prev_8y and condition_for_reset==0:
                            print ("Open P_8")
                            while opening:
                                gameDisplay.blit(pat_hd_8,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_8x, prev_8y = 0, 0
                        elif pat_9_X == prev_9x and pat_9_Y == prev_9y and condition_for_reset==0:
                            print ("Open P_9")
                            while opening:
                                gameDisplay.blit(pat_hd_9,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_9x, prev_9y = 0, 0
                        elif pat_10_X == prev_10x and pat_10_Y == prev_10y and condition_for_reset==0:
                            print ("Open P_10")
                            while opening:
                                gameDisplay.blit(pat_hd_10,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_10x, prev_10y = 0, 0
                        elif pat_11_X == prev_11x and pat_11_Y == prev_11y and condition_for_reset==0:
                            print ("Open P_11")
                            while opening:
                                gameDisplay.blit(pat_hd_11,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                            prev_11x, prev_11y = 0, 0
                        elif Exp_im_X < mouse[0] < Exp_im_X + block_width_x and Exp_im_Y < mouse[1] < Exp_im_Y + block_height_y:
                            print ("Open exp")
                            while opening:
                                gameDisplay.blit(exp_hd,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                        elif doc_1_X < mouse[0] < doc_1_X + block_width_x and doc_1_Y < mouse[1] < doc_1_Y + block_height_y:
                            print ("Open d_1")
                            while opening:
                                gameDisplay.blit(doc_hd_1,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                        elif doc_2_X < mouse[0] < doc_2_X + block_width_x and doc_2_Y < mouse[1] < doc_2_Y + block_height_y:
                            print ("Open d_2")
                            while opening:
                                gameDisplay.blit(doc_hd_2,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                        elif doc_3_X < mouse[0] < doc_3_X + block_width_x and doc_3_Y < mouse[1] < doc_3_Y + block_height_y:
                            print ("Open d3")
                            while opening:
                                gameDisplay.blit(doc_hd_3,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break

                        elif doc_4_X < mouse[0] < doc_4_X + block_width_x and doc_4_Y < mouse[1] < doc_4_Y + block_height_y:
                            print ("Open d_4")
                            while opening:
                                gameDisplay.blit(doc_hd_4,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                        elif doc_5_X < mouse[0] < doc_5_X + block_width_x and doc_5_Y < mouse[1] < doc_5_Y + block_height_y:
                            print ("Open d_5")
                            while opening:
                                gameDisplay.blit(doc_hd_5,(open_zoomX,open_zoomY)) 
                                pygame.display.update()
                                for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            pygame.quit()
                                            quit()
                                        elif event.type == pygame.MOUSEBUTTONDOWN:
                                            #print click
                                            if event.button == 1:
                                                print ("clicked")
                                                opening = False
                                                break
                        condition_for_reset==0
##                        elif b1x+b1w+bgx<pat_1_X<b1x+b1w*2+bgx and b1y<pat_1_Y<b1y+b1h:
##                            pat_1_X , pat_1_Y = b1x+b1w+bgx,b1y

                        if pat1_ == True:
                            priority == "P1"
                        if pat2_ == True:
                            priority == "P2"
                        if pat3_ == True:
                            priority == "P3"
                        if pat4_ == True:
                            priority == "P4"
                        if pat5_ == True:
                            priority == "P5"
                        if pat6_ == True:
                            priority == "P6"
                        if pat7_ == True:
                            priority == "P7"
                        if pat8_ == True:
                            priority == "P8"
                        if pat9_ == True:
                            priority == "P9"
                        if pat10_ == True:
                            priority == "P10"
                        if pat11_ == True:
                            priority == "P11"


                        pat1_ = False
                        pat2_ = False
                        pat3_ = False
                        pat4_ = False
                        pat5_ = False
                        pat6_ = False
                        pat7_ = False
                        pat8_ = False
                        pat9_ = False
                        pat10_ = False
                        pat11_ = False 

                
                elif event.type == pygame.MOUSEMOTION:
                    if pat1_drag:
                        mouse_x, mouse_y = event.pos
                        pat_1_X  = mouse_x + offset_x
                        pat_1_Y  = mouse_y + offset_y
                        priority = "P1"
                    if pat2_drag:
                        mouse_x, mouse_y = event.pos
                        pat_2_X  = mouse_x + offset_x
                        pat_2_Y  = mouse_y + offset_y
                        priority = "P2"
                    if pat3_drag:
                        mouse_x, mouse_y = event.pos
                        pat_3_X  = mouse_x + offset_x
                        pat_3_Y  = mouse_y + offset_y
                        priority = "P3"
                    if pat4_drag:
                        mouse_x, mouse_y = event.pos
                        pat_4_X  = mouse_x + offset_x
                        pat_4_Y  = mouse_y + offset_y
                        priority = "P4"
                    if pat5_drag:
                        mouse_x, mouse_y = event.pos
                        pat_5_X  = mouse_x + offset_x
                        pat_5_Y  = mouse_y + offset_y
                        priority = "P5"
                    if pat6_drag:
                        mouse_x, mouse_y = event.pos
                        pat_6_X  = mouse_x + offset_x
                        pat_6_Y  = mouse_y + offset_y
                        priority = "P6"
                    if pat7_drag:
                        mouse_x, mouse_y = event.pos
                        pat_7_X  = mouse_x + offset_x
                        pat_7_Y  = mouse_y + offset_y
                        priority = "P7"
                    if pat8_drag:
                        mouse_x, mouse_y = event.pos
                        pat_8_X  = mouse_x + offset_x
                        pat_8_Y  = mouse_y + offset_y
                        priority = "P8"
                    if pat9_drag:
                        mouse_x, mouse_y = event.pos
                        pat_9_X  = mouse_x + offset_x
                        pat_9_Y  = mouse_y + offset_y
                        priority = "P9"
                    if pat10_drag:
                        mouse_x, mouse_y = event.pos
                        pat_10_X  = mouse_x + offset_x
                        pat_10_Y  = mouse_y + offset_y
                        priority = "P10"
                    if pat11_drag:
                        mouse_x, mouse_y = event.pos
                        pat_11_X  = mouse_x + offset_x
                        pat_11_Y  = mouse_y + offset_y
                        priority = "P11"
    

                       
                        

            Video3()
            gameDisplay.blit(Layout_im,(0,0))
            text_display3(white,display_width*0.1125,display_height*0.048611,c1)
            text_display3(white,display_width*0.20692,display_height*0.048611,c2)
            text_display3(white,display_width*0.3048,display_height*0.048611,c3)
            text_display3(white,display_width*0.4,display_height*0.048611,c4)
            text_display3(white,display_width*0.49531,display_height*0.048611,c5)
            gameDisplay.blit(doc_1,(doc_1_X, doc_1_Y))
            gameDisplay.blit(doc_2,(doc_2_X, doc_2_Y))
            gameDisplay.blit(doc_3,(doc_3_X, doc_3_Y))
            gameDisplay.blit(doc_4,(doc_4_X, doc_4_Y))
            gameDisplay.blit(doc_5,(doc_5_X, doc_5_Y))
            gameDisplay.blit(Exp_im,(Exp_im_X, Exp_im_Y))
            if priority == "NONE":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P1":
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P2":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P3":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P4":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P5":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P6":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P7":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P8":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P9":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P10":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            if priority == "P11":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
            if priority == "P1":
                gameDisplay.blit(pat_1,(pat_1_X,pat_1_Y))
            if priority == "P2":
                gameDisplay.blit(pat_2,(pat_2_X,pat_2_Y))
            if priority == "P3":
                gameDisplay.blit(pat_3,(pat_3_X,pat_3_Y))
            if priority == "P4":
                gameDisplay.blit(pat_4,(pat_4_X,pat_4_Y))
            if priority == "P5":
                gameDisplay.blit(pat_5,(pat_5_X,pat_5_Y))
            if priority == "P6":
                gameDisplay.blit(pat_6,(pat_6_X,pat_6_Y))
            if priority == "P7":
                gameDisplay.blit(pat_7,(pat_7_X,pat_7_Y))
            if priority == "P8":
                gameDisplay.blit(pat_8,(pat_8_X,pat_8_Y))
            if priority == "P9":
                gameDisplay.blit(pat_9,(pat_9_X,pat_9_Y))
            if priority == "P10":
                gameDisplay.blit(pat_10,(pat_10_X,pat_10_Y))
            if priority == "P11":
                gameDisplay.blit(pat_11,(pat_11_X,pat_11_Y))
            pygame.display.flip()
 

def screen2():
    screen2 = True
    while screen2:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                
            ret, frame = camera.read()
            gameDisplay.fill([0,0,0])
            if ret is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break
            frame = np.rot90(frame)
            frame = cv2.flip( frame, 0 )
            
            frame = pygame.surfarray.make_surface(frame)
            
            gameDisplay.blit(frame, (0,0))
            pygame.display.update()
        screen2 = False
        screen3()
        

        
    

def screen1(screen1_pass_var,screen1_pass_count,screen1_pass):
    screen1 = True
    while screen1:
        background = pygame.image.load('background.png')
        background = pygame.transform.scale(background,(display_width,display_height))
        gameDisplay.blit(background,(0,0))
        login_image = pygame.image.load('keyboard_login.png')
        login_image = pygame.transform.scale(login_image,(display_width,display_height))
        gameDisplay.blit(login_image,(0,0))
        #print mouse
        for event in  pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button is 1:
                    mouse = pygame.mouse.get_pos()
                    if s1_Q_x < mouse[0] < s1_Q_x + s1_b_w and s1_Q_y < mouse[1] < s1_Q_y + s1_b_h and screen1_pass_count < 12:    
                        pygame.draw.rect(gameDisplay,red,(s1_Q_x,s1_Q_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('Q')
                        screen1_pass_count += 1
                    if s1_W_x < mouse[0] < s1_W_x + s1_b_w and s1_W_y < mouse[1] < s1_W_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_W_x,s1_W_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('W')
                        screen1_pass_count += 1
                    if s1_E_x < mouse[0] < s1_E_x + s1_b_w and s1_E_y < mouse[1] < s1_E_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_E_x,s1_E_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('E')
                        screen1_pass_count += 1
                    if s1_R_x < mouse[0] < s1_R_x + s1_b_w and s1_R_y < mouse[1] < s1_R_y + s1_b_h and screen1_pass_count < 12:    
                        pygame.draw.rect(gameDisplay,red,(s1_R_x,s1_R_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('R')
                        screen1_pass_count += 1
                    if s1_T_x < mouse[0] < s1_T_x + s1_b_w and s1_T_y < mouse[1] < s1_T_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_T_x,s1_T_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('T')
                        screen1_pass_count += 1
                    if s1_Y_x < mouse[0] < s1_Y_x + s1_b_w and s1_Y_y < mouse[1] < s1_Y_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_Y_x,s1_Y_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('Y')
                        screen1_pass_count += 1
                    if s1_U_x < mouse[0] < s1_U_x + s1_b_w and s1_U_y < mouse[1] < s1_U_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_U_x,s1_U_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('U')
                        screen1_pass_count += 1
                    if s1_I_x < mouse[0] < s1_I_x + s1_b_w and s1_I_y < mouse[1] < s1_I_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_I_x,s1_I_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('I')
                        screen1_pass_count += 1
                    if s1_O_x < mouse[0] < s1_O_x + s1_b_w and s1_O_y < mouse[1] < s1_O_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_O_x,s1_O_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('O')
                        screen1_pass_count += 1
                    if s1_P_x < mouse[0] < s1_P_x + s1_b_w and s1_P_y < mouse[1] < s1_P_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_P_x,s1_P_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('P')
                        screen1_pass_count += 1
                    if s1_A_x < mouse[0] < s1_A_x + s1_b_w and s1_A_y < mouse[1] < s1_A_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_A_x,s1_A_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('A')
                        screen1_pass_count += 1
                    if s1_S_x < mouse[0] < s1_S_x + s1_b_w and s1_S_y < mouse[1] < s1_S_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_S_x,s1_S_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('S')
                        screen1_pass_count += 1
                    if s1_D_x < mouse[0] < s1_D_x + s1_b_w and s1_D_y < mouse[1] < s1_D_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_D_x,s1_D_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('D')
                        screen1_pass_count += 1
                    if s1_F_x < mouse[0] < s1_F_x + s1_b_w and s1_F_y < mouse[1] < s1_F_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_F_x,s1_F_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('F')
                        screen1_pass_count += 1
                    if s1_G_x < mouse[0] < s1_G_x + s1_b_w and s1_G_y < mouse[1] < s1_G_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_G_x,s1_G_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('G')
                        screen1_pass_count += 1
                    if s1_H_x < mouse[0] < s1_H_x + s1_b_w and s1_H_y < mouse[1] < s1_H_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_H_x,s1_H_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('H')
                        screen1_pass_count += 1
                    if s1_J_x < mouse[0] < s1_J_x + s1_b_w and s1_J_y < mouse[1] < s1_J_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_J_x,s1_J_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('J')
                        screen1_pass_count += 1
                    if s1_K_x < mouse[0] < s1_K_x + s1_b_w and s1_K_y < mouse[1] < s1_K_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_K_x,s1_K_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('K')
                        screen1_pass_count += 1
                    if s1_L_x < mouse[0] < s1_L_x + s1_b_w and s1_L_y < mouse[1] < s1_L_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_L_x,s1_L_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('L')
                        screen1_pass_count += 1
                    if s1_Z_x < mouse[0] < s1_Z_x + s1_b_w and s1_Z_y < mouse[1] < s1_Z_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_Z_x,s1_Z_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('Z')
                        screen1_pass_count += 1
                    if s1_X_x < mouse[0] < s1_X_x + s1_b_w and s1_X_y < mouse[1] < s1_X_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_X_x,s1_X_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('X')
                        screen1_pass_count += 1
                    if s1_C_x < mouse[0] < s1_C_x + s1_b_w and s1_C_y < mouse[1] < s1_C_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_C_x,s1_C_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('C')
                        screen1_pass_count += 1
                    if s1_V_x < mouse[0] < s1_V_x + s1_b_w and s1_V_y < mouse[1] < s1_V_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_V_x,s1_V_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('V')
                        screen1_pass_count += 1
                    if s1_B_x < mouse[0] < s1_B_x + s1_b_w and s1_B_y < mouse[1] < s1_B_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_B_x,s1_B_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('B')
                        screen1_pass_count += 1
                    if s1_N_x < mouse[0] < s1_N_x + s1_b_w and s1_N_y < mouse[1] < s1_N_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_N_x,s1_N_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('N')
                        screen1_pass_count += 1
                    if s1_M_x < mouse[0] < s1_M_x + s1_b_w and s1_M_y < mouse[1] < s1_M_y + s1_b_h and screen1_pass_count < 12:
                        pygame.draw.rect(gameDisplay,red,(s1_M_x,s1_M_y,s1_b_w,s1_b_h))
                        screen1_pass_var.append('M')
                        screen1_pass_count += 1
                    if s1_backspace_x < mouse[0] < s1_backspace_x + s1_backspace_w and s1_backspace_y < mouse[1] < s1_backspace_y + s1_backspace_h and screen1_pass_count > 0:
                        pygame.draw.rect(gameDisplay,red,(s1_backspace_x,s1_backspace_y,s1_backspace_w,s1_backspace_h))
                        screen1_pass_count -= 1
                        del screen1_pass_var[screen1_pass_count]#
                    if s1_submit_x < mouse[0] < s1_submit_x + s1_submit_w and s1_submit_y < mouse[1] < s1_submit_y + s1_submit_h:
                        pygame.draw.rect(gameDisplay,red,(s1_submit_x,s1_submit_y,s1_submit_w,s1_submit_h))
                        if screen1_pass_var == screen1_pass:
                            screen1 = False
                            screen2()
                        else:
                            text_display(pure_red)
                            pygame.display.update()
                            time.sleep(1)
                            screen1_pass_count = 0
                            del screen1_pass_var[:]
                    
                    
            elif event.type == pygame.QUIT:
                pygame.quit()
                quit()
        text_display(black)
                    
        
        pygame.display.update()
        clock.tick(60)

screen1(screen1_pass_var,screen1_pass_count,screen1_pass)
#screen3()
pygame.quit()
quit()
