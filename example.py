from Anime4KPython import Anime4K
import time

if __name__ == "__main__":
    #give arguments
    anime4k = Anime4K(passes=2,strengthColor=1/3,strengthGradient=1,fastMode=False)
    #load your image
    anime4k.loadImage("your_image")
    #show basic infomation
    anime4k.showInfo()
    time_start = time.time()
    #main process
    anime4k.process()
    time_end = time.time()
    print("Total time:", time_end - time_start, "s")
    #show thr result by opencv
    anime4k.show()
    #save to disk
    anime4k.saveImage("path")
