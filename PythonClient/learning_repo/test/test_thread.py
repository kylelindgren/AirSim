# import threading
# import time
#
#
# def timer(name, delay, repeat):
#     print ("Timer: " + name + " Started")
#     while repeat > 0:
#         time.sleep(delay)
#         print (name + ": " + str(time.ctime(time.time())))
#         repeat -= 1
#     print ("Timer: " + name + " Completed")
#
# def Main():
#     t1 = threading.Thread(target=timer, args=("Timer1", 1, 5))
#     t2 = threading.Thread(target=timer, args=("Timer2", 2, 5))
#     t1.start()
#     t2.start()
#
#     print ("Main complete")
#
# if __name__ == '__main__':
#     Main()

import threading
import time

class AsyncWrite(threading.Thread):
    def __init__(self, text, out):
        threading.Thread.__init__(self)
        self.text = text
        self.out = out

    def run(self):
        # f = open(self.out, "a")
        # f.write(self.text + '\n')
        # f.close()
        time.sleep(2)
        print ("Finished Background file write to " + self.out)


def Main():
    message = input("Enter a string to store:" )
    background = AsyncWrite(message, 'out.txt')
    background.start()
    print ("The program can continue while it writes in another thread")
    print ("100 + 400 = ", 100+400)

    # background.join()
    background.run()
    background.run()
    print ("Waited until thread was complete")

if __name__ == '__main__':
    Main()
