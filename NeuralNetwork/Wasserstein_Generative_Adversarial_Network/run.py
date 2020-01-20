
import WGAN_tensor as W 
import pickle 
import numpy as np 



def main():
    epoch_d = 1
    epoch_g = 1
    batch_size = 12
    
    
    brain = W.WGAN(G_input_shape = 3, D_input_shape = 2, D_output_shape = 1, neuron = 3)
    brain.build_model()

    with open("circle_scatter.pickle", "rb") as f:
        data = pickle.load(f)
    
    data = np.array(data)

    #while True:
    noise = np.array([np.array(np.random.uniform(0, 1, size = 3), dtype = np.float64) for i in range(batch_size)])
    noise_y = [[0] for _ in range(batch_size)]
    
    gx = list(map(list, np.array(brain.generator(noise))))
    
    real_data = list(data[np.random.choice(len(data), batch_size), :])
    real_y = [[1] for _ in range(batch_size)]

    _x, _y = gx + real_data, noise_y + real_y
    batch_data = ((np.array([_x, _y]).T))
    np.random.shuffle(batch_data)
    x, y = np.array(list(batch_data.T[0])), np.array(list(batch_data.T[1]))

    for _epoch in range(epoch_d):
        brain.train_disciminator(x, y)
    print("discriminator training finished")

    for _epoch in range(epoch_g):
        brain.train_generator(noise, np.array(noise_y))
    print("generator training finished")
    
    
    

        

if __name__ == "__main__":
    main()