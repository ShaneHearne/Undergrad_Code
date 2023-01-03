import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from matplotlib.ticker import MaxNLocator
from skimage.transform import resize

#Importing packages

states = 4
unburnable, burnable, burning, burnt= range (states)
colors= ListedColormap (['white','green','red','black'])
labels=['unburnable', 'burnable', 'burning', 'burnt']

#BUILDING MODEL AND STARTING FIRES
def model_setup(shape=(10,20), seed=None, fires=None, unburnables=None, image=None):
    global model
    m, n = shape
    model=np.zeros((m,n), int)
    model[1:-1,1:-1]=burnable
    
    if seed is not None:
        np.random.seed(seed)
        
    if unburnables is None:
        pass
    
    elif type(unburnables)==int:
        count=0
        while count<unburnables:
            r, c = np.random.randint(1,m-1),np.random.randint(1,n-1)
            if model[r,c]==burnable:
                model[r,c] = unburnable
                count+=1
                
    else:
        for r,c in unburnables:
            model[r,c] = unburnable
    
    if fires is None:
        model[m//2,n//2] = burning
        
    elif type(fires)==int:
        count=0
        while count<fires:
            r, c = np.random.randint(1,m-1),np.random.randint(1,n-1)
            if (model[r,c]==burnable):
                model[r,c] = burning
                count+=1
        
    else:
        for r,c in fires:
            model[r,c] = burning
            
            
    if image is None:
        pass
    
    if image is True:
        forest = plt.imread('island.png')
     
        plt.imshow(forest[:,:,1] < 0.4,vmin=0, vmax=4, cmap=colors)
        
        model = (forest[:,:,1] < 0.4)   

#DISPLAYING MODEL
def model_display(duration=0.01, save=False, t=0):
     if duration <= 0 and not save: 
         return
     plt.imshow(model, vmin=0, vmax=states-1, cmap=colors)
     if save: 
         plt.savefig(f'Fire_{t:05d}.pdf', bbox_inches='tight')
     plt.pause(duration)


#DOING SIMULATION
def model_simulate(t_max=None,display=False,pr_burnable_to_burning=int,pr_burning_to_burnt=int):
        global current, model
        m,n= model.shape
        if t_max is None:
            t_max = max(*model.shape)**2
            
        for t in range(t_max):
            current = model.copy()     
            for r in range(1,m-1):
                for c in range(1,n-1): 
                    if current[r,c]==burnable:
                            if (np.random.uniform() < pr_burnable_to_burning*neighbours(r,c,burning)):
                                model[r,c] = burning
                    elif current[r,c]==burning:
                        if (np.random.random() < pr_burning_to_burnt): 
                            model[r,c]=burnt
                            
            if display: 
                model_display(0.001,t,False)
            
                
    
            if np.sum(model==burning)==0:
                return True
    
        return False
           
def model_summary(show_summary=True, message=''):
    summary= {}
    if show_summary is False:
        for state, label in enumerate(labels):
            summary[label] = np.sum(model==state) 
        if show_summary:
            print(message)
    if show_summary is True:
        for state, label in enumerate(labels):
            summary[label] = np.sum(model==state) 
        if show_summary:
            print(message)
        for label in labels:
            print('\t %5d %s cells' % (summary[label],label))
        
    
        
    return summary
       
    
def neighbours(r,c,burning):
    direct = int(current[r-1,c]==burning) + int(current[r+1,c]==burning) + int(current[r,c-1]==burning) + int(current[r,c+1]==burning)
    diagonal = int(current[r-1,c-1]==burning) + int(current[r+1,c+1]==burning) + int(current[r+1,c-1]==burning) + int(current[r-1,c+1]==burning)
    return (direct+diagonal/2)/6

def model_damage(start,end):
     assert start['burnable']+start['burning']+start['burnt']==end['burnable']+end['burning']+end['burnt']
     damage = 1 - end['burnable']/start['burnable']
     return damage
 
    
    
         
def Figure1(iterations):
    FD = []
    for n in range(iterations):
        model_setup(shape=(10,20),fires=1)
        start = model_summary(show_summary=False)
        model_simulate(pr_burnable_to_burning=1, pr_burning_to_burnt=0.2,display=False)
        end = model_summary(show_summary=False)
        FireDmg=(round(100*model_damage(start,end),0))
        FD.append(FireDmg)
    plt.ylabel('Relative Frequency')
    plt.xlabel('Fire Damage %')
    plt.savefig("Figure 1")
    plt.title('Distribution of fire damage due to noise')
    plt.hist(FD,bins=100,density=True)
    
    
def Figure2(samples, sample_size):
    FD_data_avg_array = []
    for m in range (samples):
        FD_data = []
        for n in range(sample_size):
            model_setup(shape=(10,20),fires=1)
            start = model_summary(show_summary=False)
            model_simulate(pr_burnable_to_burning=1, pr_burning_to_burnt=0.2,display=False)
            end = model_summary(show_summary=False)
            FD_data.append(round(100*model_damage(start,end),0))
        
        FD_data_avg = (sum(FD_data)/len(FD_data))
        FD_data_avg_array.append(FD_data_avg)
        
    x = np.linspace(1,samples,samples)
    plt.figure(figsize=(5,10))
    ax=plt.axes()
    ax.set_facecolor('lavender')
    plt.grid(color='white')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0,100)
    plt.xlim(1,samples)
    plt.xlabel("Samples")
    plt.ylabel("Fire Damage %")
    plt.title("Sample Size: %i" % sample_size)
    plt.savefig("Figure 2")
    plt.plot(x,FD_data_avg_array)
    plt.show()
    



def Figure3(sample_size) :
    size = [5, 10, 15, 20, 30, 50] 
    FD_data_avg_array = []
    for m in size:
        FD_data = []
        for n in range(sample_size):
            model_setup(shape=(m,m),fires=1)
            start = model_summary(show_summary=False)
            model_simulate(pr_burnable_to_burning=0.5, pr_burning_to_burnt=0.1,display=False)
            end = model_summary(show_summary=False)
            FD_data.append(round(model_damage(start,end),0))
        
        FD_data_avg = (sum(FD_data)/len(FD_data))
        FD_data_avg_array.append(FD_data_avg)
        
    plt.ylabel("Fire Damage")
    plt.xlabel("Grid Size (m)")
    plt.title("Damage due to grid size")
    plt.savefig("Figure 3")
    plt.plot(size,FD_data_avg_array, "bo")


def Figure4(iterations):
    increments=10
    spaces=np.linspace(0,1,increments)
    data = np.zeros((increments,increments))
    for i, pr_burnable_to_burning in enumerate(spaces): 
            for j, pr_burning_to_burnt in enumerate(spaces):
                for k in range(iterations):
                    model_setup()
                    start = model_summary(show_summary=False)
                    model_simulate(pr_burnable_to_burning=pr_burnable_to_burning,pr_burning_to_burnt=pr_burning_to_burnt,display=False)
                    end=model_summary(show_summary=False)
                    data[i,j]+= model_damage(start,end)
    data = 100*data/iterations
    increments_avg = increments // 2
    data_resized = resize(data, (increments_avg, increments_avg))
    
    plt.figure(figsize=(8,8))    
    contours = plt.contour(data_resized, extent=[0,1,0,1], levels=[1,2,5,25,50,75,95])    
    plt.clabel(contours, inline=True, fontsize=12, fmt="%.0f%%")
    plt.ylabel("Spread Rate\nPr(burnable \u2192 burning)")
    plt.xlabel("Burn Rate\nPr(burning \u2192 burnt)")
    plt.title("Damage dependence on on fire spread and burn rates")
    plt.savefig("Figure 4")
    plt.show()

 




def Figure5a(n_points, sample_size):
    pr_set = [0.01, 0.21, 0.42, 0.62, 0.83]
    Averaged_damage = []
    pr_range = np.linspace(0,1,n_points)
    #for a in pr_set:
    for s in pr_set:
        
        for b in pr_range:
            FD_data = []
            for n in range(sample_size):
                
                model_setup(shape=(10,20),fires=1,)
                start = model_summary(show_summary=False)
                model_simulate(pr_burnable_to_burning = s, pr_burning_to_burnt = b,display=False)
                end = model_summary(show_summary=False)
                FD_data.append(round(100*model_damage(start,end),0))
            
            FD_data_avg = (sum(FD_data)/len(FD_data))
            Averaged_damage.append(FD_data_avg)
            
        
    spread_rate_1 = Averaged_damage[0:n_points]
    spread_rate_21 = Averaged_damage[n_points:2*n_points]
    spread_rate_42 = Averaged_damage[2*n_points:3*n_points]
    spread_rate_62 = Averaged_damage[3*n_points:4*n_points]
    spread_rate_83 = Averaged_damage[4*n_points:5*n_points]

#This is probably not the best way to do this, but its best I could do. 
#After it has been averaged, the data is stored in Averaged_damage    
    

    plt.ylabel("Fire Damage %")
    plt.xlabel("Burn Rate\nPr(burning \u2192 burnt)")
    plt.title("Damage due to burn rate")
    plt.savefig("Figure 5a")
    plt.plot(pr_range, spread_rate_1, pr_range, spread_rate_21, pr_range, spread_rate_42,pr_range, spread_rate_62,pr_range, spread_rate_83)
    plt.show()






def Figure5b(n_points, sample_size):
    pr_set = [0.01, 0.09, 0.21, 0.42, 0.62, 0.83]
    Averaged_damage = []
    pr_range = np.linspace(0,1,n_points)
    #for a in pr_set:
    for s in pr_set:
        for b in pr_range:
            FD_data = []
            for n in range(sample_size):
                
                model_setup(shape=(10,20),fires=1,)
                start = model_summary(show_summary=False)
                model_simulate(pr_burnable_to_burning = b, pr_burning_to_burnt = s ,display=False)
                end = model_summary(show_summary=False)
                FD_data.append(round(100*model_damage(start,end),0))
            
            FD_data_avg = (sum(FD_data)/len(FD_data))
            Averaged_damage.append(FD_data_avg)
            
        
    spread_rate_1 = Averaged_damage[0:n_points]
    spread_rate_21 = Averaged_damage[n_points:2*n_points]
    spread_rate_42 = Averaged_damage[2*n_points:3*n_points]
    spread_rate_62 = Averaged_damage[3*n_points:4*n_points]
    spread_rate_83 = Averaged_damage[4*n_points:5*n_points]
    
    
    plt.ylabel("Fire Damage %")
    plt.xlabel("Spread Rate\nPr(burnable \u2192 burning)")
    plt.title("Damage due to spread rate")
    plt.savefig("Figure 5b")
    plt.plot(pr_range, spread_rate_1, pr_range, spread_rate_21, pr_range, spread_rate_42,pr_range, spread_rate_62,pr_range, spread_rate_83)







def Figure6(sample_size):
 
    damp_array = [0,5,10,15,20,30,40,50]
    FD_data_avg_array = []
    for k in damp_array:
        FD_data = []
        for m in range(sample_size):
            model_setup(shape=(10,20),fires=1,unburnables=int(k))
            start = model_summary(show_summary=False)
            model_simulate(display=False, pr_burnable_to_burning=1, pr_burning_to_burnt=0.2)
            end = model_summary(show_summary=False)
            FD = int(round(model_damage(start,end),0))
            FD_data.append(FD)
        FD_data_avg = (sum(FD_data)/len(FD_data))
        FD_data_avg_array.append(FD_data_avg)
    
    
    ax=plt.axes()
    ax.set_facecolor('lavender')
    plt.grid(color='white')
    plt.xlabel("Grid size (m)")
    plt.ylabel("Fire Damage")
    plt.title("Damage with respect to unburnable cell density")
    plt.plot(damp_array,FD_data_avg_array,'bo')
    plt.savefig("Figure 6")



def Figure7():
    forest = plt.imread('island.png')
    print(forest.shape)
    plt.imshow(forest)
    plt.title("Satellite image of a real forest.")
    plt.savefig("Figure 7")
    plt.show()
    

def Figure8():
    forest = plt.imread('island.png')
    print(forest.shape)
    plt.imshow(forest[:,:,1] < 0.4,vmin=0, vmax=4, cmap=colors) 
    plt.title("Image with unburnable/burnable cells identified")
    plt.savefig("Figure 8")
    

def Graph9():
    forest = plt.imread('island.png')
    plt.imshow(forest[:,:,1] < 0.4, vmin=0, vmax=4, cmap=colors)
    model_setup(shape=(195,304),fires=10,image=True)
    start = model_summary(show_summary=False)
    model_simulate(display=False, pr_burnable_to_burning=1, pr_burning_to_burnt=0.2)
    end = model_summary(show_summary=False)



Figure5a(10,20)