# -*- coding:utf8 -*-
import Tkinter as tkinter
import tkMessageBox
# from tkinter.filedialog import askdirectory
import tkFileDialog
from SitupPose import SitupPose
from BasketPose import BasketPose
from PushupPose import PushupPose
import cv2
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
import numpy as np
import pickle


def print_selection():
    label_mode.config(text=mode.get())


def selectPath():
    path_ = tkFileDialog.askopenfilename()
    path.set(path_)
    label_test.config(text=path.get())

def start():
    # a = SitupPose("/home/zhu/Desktop/1.gif")
    # a = BasketPose("/home/zhu/Desktop/2.gif")
    # a = PushupPose("/home/zhu/Desktop/3.gif")
    # if path=="nothing":
    #     print('error, choose the path!')
    #     exit(0)

    if mode.get()=="basketball":
        a = BasketPose(path.get())
    if mode.get()=="pushup":
        a = PushupPose(path.get())
    if mode.get()=="situp":
        a = SitupPose(path.get())

    ret = True
    while ret is True:
        ret, img, prams = a.process()
        if ret is False:
            break
        # cv2.imshow("output", img)
        # cv2.waitKey(1)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)
        top.update()

        # label_img.after(150,imgtk)
        text=''
        for index in prams:
           # text.append(index+':'+str(prams[index]))
           # text.append('\n')
            text=text+index+':'+str(prams[index])+'\n'
        label_angel.config(text=text)
    print a.get_time_seq_pram()
    print a.eval()[0]
    zhexiaotu=ImageTk.PhotoImage(file="/home/zhu/code/openpose/build/examples/tutorial_api_python/output"+"/"+mode.get()+".jpg")
    label_zhexiaotu.imgtk=zhexiaotu
    label_zhexiaotu.configure(image=zhexiaotu)
    top.update()


top = tkinter.Tk()
top.title("火柴人科技有限公司")
#top.geometry('400x300')

# label_angel = tkinter.Label(top, text="there put the angel info")
# button_play = tkinter.Button(top, text="start!",command=start)
# label_mode = tkinter.Label(top)
# label_img=tkinter.Label(top)

mode = tkinter.StringVar()
r1 = tkinter.Radiobutton(top, text='basketball',
                         variable=mode, value='basketball',
                         command=print_selection)
r2 = tkinter.Radiobutton(top, text='situp',
                         variable=mode, value='situp',
                         command=print_selection)
r3 = tkinter.Radiobutton(top, text='pushup',
                         variable=mode, value='pushup',
                         command=print_selection)

path = tkinter.StringVar()
button_file = tkinter.Button(top, text="choose", command=selectPath)

label_test = tkinter.Label(top)

label_angel = tkinter.Label(top)
button_play = tkinter.Button(top, text="start!",command=start)
label_mode = tkinter.Label(top)
label_img=tkinter.Label(top)
label_zhexiaotu=tkinter.Label(top)

label_img.grid(row=2,column=1,columnspan=3,sticky='W')
#label_mode.grid(row=1, column=2, sticky='W')
label_angel.grid(row=2, column=5, sticky='W')
r1.grid(row=3, column=1, sticky='W')
r2.grid(row=4, column=1, sticky='W')
r3.grid(row=5, column=1, sticky='W')
button_play.grid(row=4, column=2, sticky='W')
button_file.grid(row=4, column=3, sticky='W')
label_zhexiaotu.grid(row=3,column=4,rowspan=3,columnspan=3,sticky='W')
# label_test.grid(row=5)


top.mainloop()
