from tkinter import *
import numpy as np
from neural_network import *
import matplotlib.pyplot as plt

class Gui:
    
    def __init__(self, master):
        
        # Layout and frames
        master.geometry('410x330')
        border_color='white'
        input_frame = Frame(master, highlightbackground=border_color, highlightthickness=2)
        input_frame.grid(row=2, column=0)
        feedback_frame = Frame(master, highlightbackground=border_color, highlightthickness=2)
        feedback_frame.grid(row=2, column=2)
        
        # Empty frames/labels for layout
        empty_frame_1 = Frame(master, highlightbackground=border_color, highlightthickness=2)
        empty_frame_1.grid(row=2, column=1)
        empty_label_1 = Label(empty_frame_1, text=' ', font=("Helvetica", 30))
        empty_label_1.pack()
        
        # Buttons
        self.button_reset = Button(input_frame, text='Reset', width=10, command=self.reset)
        self.button_reset.pack(side=BOTTOM)
        self.button_recognize = Button(input_frame, text='Recognize!', width=10, command=self.run_nn)
        self.button_recognize.pack(side=BOTTOM)

        # Drawing field
        sub_1 = Label(input_frame, text='Write your digit!', font=("Helvetica", 15))
        sub_1.pack(side=TOP)
        self.drawing_field = Canvas(input_frame, height=250, width=250, bg='white', cursor='cross', 
                                    highlightbackground="black", highlightthickness=2)
        self.drawing_field.pack() 
        self.drawing_field.bind("<Motion>", self.tell_me_where_you_are)
        self.drawing_field.bind("<B1-Motion>", self.draw_from_where_you_are)
        
        # Feedback field
        sub_2 = Label(feedback_frame, text='Recognized as...', font=("Helvetica", 15))
        sub_2.pack(side=TOP)
        self.prediction_field = Text(feedback_frame, height=1, width=1, font=("Helvetica", 50), bg='light grey')
        self.prediction_field.pack(side=TOP)
        
        sub_3 = Label(feedback_frame, text='Confidence...', font=("Helvetica", 15))
        sub_3.pack(side=TOP)
        self.confidence_field = Text(feedback_frame, height=1, width=4, font=("Helvetica", 50), bg='light grey')
        self.confidence_field.pack(side=TOP)
        
        sub_4 = Label(feedback_frame, text='Alternatives...', font=("Helvetica", 15))
        sub_4.pack(side=TOP)
        self.alternative_field = Text(feedback_frame, height=1, width=1, font=("Helvetica", 50), bg='light grey')
        self.alternative_field.pack(side=TOP)

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        self.x = event.x
        self.y = event.y
        self.drawing_field.create_polygon(self.previous_x, self.previous_y, self.x, self.y, 
                                width=20, outline='black')    
        self.previous_x = self.x
        self.previous_y = self.y
        
    def run_nn(self):
        output = n.run(train_images[0]).T[0]
        self.prediction = np.argmax(output)
        self.confidence = np.max(output)
        self.alternative = np.argsort(output)[-2]
        self.prediction_field.insert(END, str(self.prediction))
        self.confidence_field.insert(END, '%.0f%%' %(self.confidence*100))
        if self.confidence < 0.8:
            self.alternative_field.insert(END, str(self.alternative))
        else:
            self.alternative_field.insert(END, '/')
        
    def reset(self):
        self.prediction_field.delete(1.0,END)
        self.confidence_field.delete(1.0,END)
        self.alternative_field.delete(1.0,END)
        self.drawing_field.delete('all')

root = Tk()
b = Gui(root)
root.mainloop()