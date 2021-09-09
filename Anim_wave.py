from matplotlib import pyplot as plt  
from matplotlib import animation  
import matplotlib.image as mgimg
import numpy as np

#set up the figure
fig = plt.figure()
ax = plt.gca()

#initialization of animation, plot array of zeros 
def init():
    imobj.set_data(np.zeros((200, 200)))

    return  imobj,

def animate(i):
    ## Read in picture
    #fname ="/home/nick/Python_code/gitnonlinearcq/data/wave_images/Screen_n{}.png".format(i)
    fname ="data/wave_images/Screen_n{}.png".format(i)
#    fname ="/home/nick/Python_code/Anim/wave_imag/GIBCe2_height02_outer_magnet_n{}.png".format(i)
    ## here I use [-1::-1], to invert the array
    # IOtherwise it plots up-side down
    img = mgimg.imread(fname)[-1::-1]
    imobj.set_data(img)

    return  imobj,


## create an AxesImage object
imobj = ax.imshow( np.zeros((200, 200)), origin='lower', alpha=1.0, zorder=1, aspect=1 )


anim = animation.FuncAnimation(fig, animate, init_func=init, repeat = True,
                               frames=range(0,30), interval=20, blit=True, repeat_delay=500)
anim.save('NonlinearSphere2.mp4',fps=40,dpi=300,bitrate=-1)
plt.show()
