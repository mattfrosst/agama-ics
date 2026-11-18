import numpy as np

def main(length):
    with open('cmd_computeorbits.txt','w') as f:
        for i in range(length):
            line = './Submit_computeorbits.sh '+str(i)
            f.write(line); f.write('\n');


if __name__ == '__main__':
    length = 2#1042
    main(length)

