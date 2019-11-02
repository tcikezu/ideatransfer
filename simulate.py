""" simulate.py
-----------------------
This file creates a community object, defined in test.py, and runs to see how ideas
evolve over some number of iterations."""

import transfer
from community import *
import  matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


gamma = 0.008
T = 200
X = community(300,3)
x,y,z = [], [], []

fn = 'probabilistic merging'
for t in range(T):
    #transfer.deterministicMerge(X, gamma)
    transfer.probabilisticMerge(X,gamma)
    x.append(X.allIdeas[:,0])
    y.append(X.allIdeas[:,1])
    z.append(X.allIdeas[:,2])


fps = 40 

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
sct, = ax.plot([],[],[],"o",markersize=2)

def update(ifrm, xa, ya, za):
    sct.set_data(xa[ifrm], ya[ifrm])
    sct.set_3d_properties(za[ifrm])
    ax.set_xlabel("frame: %d" % (ifrm))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ani = animation.FuncAnimation(fig, update, T, fargs=(x,y,z),interval = T/fps )


#ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
ani.save(fn+'.gif',writer='imagemagick',fps=fps)

plt.show()


