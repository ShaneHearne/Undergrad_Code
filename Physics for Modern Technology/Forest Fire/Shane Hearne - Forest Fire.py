import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from matplotlib.ticker import MaxNLocator
from skimage.transform import resize
import matplotlib.image as mpimg
#Importing packages

#Setting up 4 states and assigning a label and colour to each
states = 4
unburnable, burnable, burning, burnt= range (states)
colors= ListedColormap (['white','green','red','black'])
labels=['unburnable', 'burnable', 'burning', 'burnt']


#BUILDING MODEL AND STARTING FIRES
def model_setup(shape=(10,20), seed=None, fires=None, unburnables=None, image=None):
#Creating grid of zero states, using m and n to define the shape of grid
    
    global model
    m, n = shape
    model=np.zeros((m,n), int)
    model[1:-1,1:-1]=burnable
    
#This code is fore figure 9, if image is set to none, it build normal model, if image is True, the image is converted to a model                        
    if image is None:
            m, n = shape
            model=np.zeros((m,n), int)
            model[1:-1,1:-1]=burnable 
        
    if image is True:
            m,n = forest.shape
            model = forest.copy()
            model[0] = model[-1] = unburnable
            model[-1] = model[1] = unburnable
            
    
#If seed is set to anything other then none, a seed is generated    
    if seed is not None:
        np.random.seed(seed)

#This code allows you to set number of unbunable cells, or to set exactly where these unburnable cells are created       
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


#This code allows you to set number of fires, or to set exactly where these fires are created    
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



            
#DISPLAYING MODEL
def model_display(duration=0.01, save=False, t=0):
     if duration <= 0 and not save: 
         return
     plt.imshow(model, vmin=0, vmax=states-1, cmap=colors)
     if save: 
         plt.savefig(f'Fire_{t:05d}.pdf', bbox_inches='tight')
     plt.pause(duration)


#DOING SIMULATION - This is where the model we have setup is simulated. We can set both probabilities
#This code creates a copy of model and cycles through each row and column and applies our probabilities 
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
    

#This code displays a start and end summary of our model, used with model_damage it allows us to compare the fire dmage       
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
       
#This code establishes what we mean by neighbours, here we take 8 surrounding cells to be available   
def neighbours(r,c,burning):
    direct = int(current[r-1,c]==burning) + int(current[r+1,c]==burning) + int(current[r,c-1]==burning) + int(current[r,c+1]==burning)
    diagonal = int(current[r-1,c-1]==burning) + int(current[r+1,c+1]==burning) + int(current[r+1,c-1]==burning) + int(current[r-1,c+1]==burning)
    return (direct+diagonal/2)/6


#This allows us to view the percentage damage that the fire has caused
def model_damage(start,end):
     assert start['burnable']+start['burning']+start['burnt']==end['burnable']+end['burning']+end['burnt']
     damage = 1 - end['burnable']/start['burnable']
     return damage
 
    
    
#Figure one demenstates the effect of noise by plotting a histogram of fire damages, showing extreme variation in results
#It shows that the most likely scenarios are that the fire dies out straight away or, if it gets going, it will probably do a high percentage damage.          
#FD is an array of fire damages
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
#    plt.savefig("Figure 1 - Distribution of fire damage under above simulation parameters")
    
  
#This code shows the effect of sample size on the average damage. It allows us to choose which sample size to use
#FD_data stores each sample and is used to store the data before it averaged
# FD_data_avg_array is introduced to store the averaged data

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
    plt.plot(x,FD_data_avg_array)
#    plt.savefig("Figure 2 - Sample size %i" % sample_size)
    
    


#Size is an array of grid sizes in which we want to test. We me to each of these values and then average across a select sample size
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
    plt.plot(size,FD_data_avg_array, "bo")
    plt.savefig("Figure 3")


#This graph is very important as it allows us to see the effect of varying probabilites for both burn rate and spread rate, and how they effect each other.
#This is the first time I used enumerate, I should have used this all along but was unsure how to impliment it until shown in class
#Instead of creating an emtpy array and apending the values, an empty array of zeros is created, similar to the model setup
#Increments is set to 10 so we have 10 evenly spaced intervals
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
    

 



#I realise that there is probably a more simple wayto achieve this graph but I could not figure it out. This ad-hoc method does produces the desired result 
#At the time of writing this code, i was not familiar with enumerate, this could probably be used as a more attractiive method    
#This code works by selecting each spread rate probabiliy from the pr_set and varying the the burn rate. Averaging is applied and is controlled by sample size
#n_points selects the amount of increments that the burn rate will go through between 0 and 1
#That means that each value of spread rate will have n_point averaged values stored, so I used this fact to take the corresponding n_points from average data

def Figure5a(n_points, sample_size):
    pr_set = [0.01, 0.21, 0.42, 0.62, 0.83]
    Averaged_damage = []
    pr_range = np.linspace(0,1,n_points)
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


    plt.ylabel("Fire Damage %")
    plt.xlabel("Burn Rate\nPr(burning \u2192 burnt)")
    plt.title("Damage due to burn rate")
    ax = plt.subplot(111)
    ax.plot(pr_range, spread_rate_1, label='Spread rate = 1%')
    ax.plot(pr_range, spread_rate_21, label='Spread rate = 21%')
    ax.plot(pr_range, spread_rate_42, label='Spread rate = 42%')
    ax.plot(pr_range, spread_rate_62, label='Spread rate = 62%')
    ax.plot(pr_range, spread_rate_83, label='Spread rate = 83%')
    ax.legend()
#    plt.savefig("Figure 5a")
 




#This code is fairly similar to 5a but burn rate and spread rate are reversed
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
            
    print(Averaged_damage)    
    burn_rate_1 = Averaged_damage[0:n_points]
    burn_rate_9 = Averaged_damage[n_points:2*n_points]
    burn_rate_21 = Averaged_damage[2*n_points:3*n_points]
    burn_rate_42 = Averaged_damage[3*n_points:4*n_points]
    burn_rate_62 = Averaged_damage[4*n_points:5*n_points]
    burn_rate_83 = Averaged_damage[5*n_points:6*n_points]
    
    
    plt.ylabel("Fire Damage %")
    plt.xlabel("Spread Rate\nPr(burnable \u2192 burning)")
    plt.title("Damage due to spread rate")
    ax = plt.subplot(111)
    ax.plot(pr_range, burn_rate_1, label='Burn rate = 1%')
    ax.plot(pr_range, burn_rate_9, label='Burn rate = 9%')
    ax.plot(pr_range, burn_rate_21, label='Burn rate = 21%')
    ax.plot(pr_range, burn_rate_42, label='Burn rate = 42%')
    ax.plot(pr_range, burn_rate_62, label='Burn rate = 62%')
    ax.plot(pr_range, burn_rate_83, label='Burn rate = 83%')
    ax.legend()
#    plt.savefig("Figure 5b")
  



#This graph shows the effect of adding a varying amount of unburnable cells. Each value in the array is averaged.
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
#    plt.savefig("Figure 6")


#This code just loads in the image and plots it
def Figure7():
    forest = plt.imread('island.png')
    print(forest.shape)
    plt.imshow(forest)
    plt.title("Satellite image of a real forest.")
#    plt.savefig("Figure 7")
    plt.show()
    
#By applying a treshold of 4 to the green values of the RBGA figure 8 can be recreated
def Figure8():
    forest = plt.imread('island.png')
    print(forest.shape)
    plt.imshow(forest[:,:,1] < 0.4,vmin=0, vmax=4, cmap=colors) 
    plt.title("Image with unburnable/burnable cells identified")
#    plt.savefig("Figure 8")
    

#This code takes the forest image and converts it a model. 10 fires are then set off in this model and the before and after are plotted
#Additonal code was added to teh setup function to allow this function to work
def Figure9():
    global forest
    forest=mpimg.imread('island.png')
    forest = (forest[:,:,1] < 0.4)
    forest=np.asarray(forest, dtype='int')
    model_setup(shape=(forest.shape),fires=10,image=True)
    plt.title("Figure 9 - Initial State")
    model_display()
    plt.savefig("Figure 9 - Initial State")
    print('At start there are:')
    start = model_summary(show_summary=True)
    model_simulate(pr_burnable_to_burning=1, pr_burning_to_burnt=0.3)
    plt.title("Figure 9 - End State")
    model_display()
    print('At end there are:')
    end = model_summary(show_summary=True)
#    plt.savefig("Figure 9 - End State")
    print("Fire damage was %.1f%%" % (100*model_damage(start,end)))
    
