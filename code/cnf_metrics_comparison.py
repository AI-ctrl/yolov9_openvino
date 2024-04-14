
validation metrics

onnx

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/val/labels.cache... 50 images, 0 backgrounds, 0 corrupt: 100% 50/50 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 50/50 [03:55<00:00,  4.72s/it]
                   all         50         82      0.323      0.521      0.346       0.26
                  vest         50          2          0          0     0.0373     0.0261
    short sleeve dress         50          1      0.089          1      0.142      0.128
      short sleeve top         50         24      0.523      0.867      0.773      0.595
            vest dress         50          6      0.229        0.5       0.28       0.19
              trousers         50         14      0.425      0.929      0.793      0.556
       long sleeve top         50          6      0.144      0.833      0.413      0.375
                shorts         50          9      0.344      0.333      0.435       0.32
                 skirt         50         10      0.253        0.6       0.36      0.276
     long sleeve dress         50          6      0.184      0.333      0.216      0.184
   long sleeve outwear         50          3      0.367      0.333      0.236      0.165
  short sleeve outwear         50          1          1          0      0.124     0.0498
Speed: 1.5ms pre-process, 4687.3ms inference, 3.2ms NMS per image at shape (1, 3, 640, 640)





best.pt 

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/val/labels.cache... 50 images, 0 backgrounds, 0 corrupt: 100% 50/50 [00:00<?, ?it/s]
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:19<00:00,  9.75s/it]
                   all         50         82      0.213        0.5      0.335      0.242
                  vest         50          2          0          0     0.0501     0.0337
    short sleeve dress         50          1     0.0814          1      0.142      0.114
      short sleeve top         50         24       0.51      0.833      0.764      0.546
            vest dress         50          6      0.342      0.667      0.433      0.355
              trousers         50         14       0.45      0.786       0.66       0.45
       long sleeve top         50          6      0.148      0.833      0.387      0.379
                shorts         50          9      0.165      0.111      0.264      0.139
                 skirt         50         10      0.277        0.6      0.371      0.274
     long sleeve dress         50          6      0.154      0.333      0.203      0.162
   long sleeve outwear         50          3      0.219      0.333       0.41      0.209
  short sleeve outwear         50          1          0          0          0          0
Speed: 0.2ms pre-process, 63.9ms inference, 18.1ms NMS per image at shape (32, 3, 640, 640)



openvino

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/val/labels.cache... 50 images, 0 backgrounds, 0 corrupt: 100% 50/50 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 50/50 [01:57<00:00,  2.36s/it]
                   all         50         82      0.327      0.521      0.351      0.253
                  vest         50          2          0          0     0.0361     0.0262
    short sleeve dress         50          1     0.0896          1      0.166      0.149
      short sleeve top         50         24      0.498       0.87      0.766      0.553
            vest dress         50          6       0.22        0.5      0.276      0.186
              trousers         50         14       0.42      0.929      0.787      0.516
       long sleeve top         50          6      0.135      0.833      0.406      0.367
                shorts         50          9      0.378      0.333       0.45      0.315
                 skirt         50         10      0.278        0.6      0.362      0.275
     long sleeve dress         50          6      0.187      0.333      0.193      0.155
   long sleeve outwear         50          3      0.387      0.333      0.256      0.171
  short sleeve outwear         50          1          1          0      0.166     0.0663
Speed: 1.1ms pre-process, 2318.2ms inference, 13.0ms NMS per image at shape (1, 3, 640, 640)




openvino_int8

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/val/labels.cache... 50 images, 0 backgrounds, 0 corrupt: 100% 50/50 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 50/50 [03:11<00:00,  3.84s/it]
                   all         50         82      0.326      0.522      0.346      0.261
                  vest         50          2          0          0     0.0374     0.0262
    short sleeve dress         50          1     0.0897          1      0.142      0.128
      short sleeve top         50         24      0.525      0.875      0.773      0.596
            vest dress         50          6      0.229        0.5       0.28       0.19
              trousers         50         14      0.429      0.929      0.793      0.556
       long sleeve top         50          6      0.147      0.833      0.413      0.375
                shorts         50          9      0.348      0.333      0.435       0.32
                 skirt         50         10      0.259        0.6       0.36      0.276
     long sleeve dress         50          6      0.184      0.333      0.216      0.184
   long sleeve outwear         50          3      0.374      0.333      0.236      0.165
  short sleeve outwear         50          1          1          0      0.124     0.0498
Speed: 0.6ms pre-process, 3798.8ms inference, 13.0ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/val/exp20



















Testing metrics 

onnx

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/test/labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100% 100/100 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 100/100 [06:44<00:00,  4.05s/it]
                   all        100        168      0.367      0.377      0.365      0.267
           sling dress        100          3          0          0      0.206      0.167
                  vest        100          5      0.398        0.2      0.179     0.0957
    short sleeve dress        100          8      0.212      0.306      0.197      0.158
      short sleeve top        100         42      0.566      0.738       0.65      0.552
            vest dress        100         11      0.208      0.364      0.359        0.3
              trousers        100         32      0.518      0.531      0.565      0.385
       long sleeve top        100         20      0.359        0.7      0.386      0.223
                shorts        100          9      0.418      0.222      0.294      0.198
                 skirt        100         25       0.52       0.64      0.542      0.388
     long sleeve dress        100          8      0.478       0.25       0.27      0.238
   long sleeve outwear        100          5      0.358        0.2      0.369      0.235
Speed: 0.5ms pre-process, 4014.7ms inference, 6.9ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/val/exp10



pt 
val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/test/labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100% 100/100 [00:00<?, ?it/s]
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:10<00:00,  2.67s/it]
                   all        100        168      0.292      0.535      0.383      0.279
           sling dress        100          3      0.284      0.333      0.251      0.206
                  vest        100          5       0.29      0.335      0.323      0.166
    short sleeve dress        100          8      0.155        0.5      0.206      0.173
      short sleeve top        100         42      0.493       0.81      0.704      0.556
            vest dress        100         11      0.178      0.455      0.346       0.29
              trousers        100         32      0.471      0.688      0.629      0.434
       long sleeve top        100         20      0.303      0.849      0.367       0.24
                shorts        100          9      0.155      0.222      0.211      0.136
                 skirt        100         25      0.421       0.64      0.456      0.327
     long sleeve dress        100          8      0.152       0.25      0.204      0.169
   long sleeve outwear        100          5      0.307        0.8      0.513      0.377
Speed: 0.3ms pre-process, 70.9ms inference, 9.4ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp14


openvino - normal
val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/test/labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100% 100/100 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 100/100 [06:28<00:00,  3.88s/it]
                   all        100        168      0.367      0.377      0.364      0.267
           sling dress        100          3          0          0      0.206      0.167
                  vest        100          5      0.398        0.2      0.179     0.0957
    short sleeve dress        100          8      0.212      0.306      0.198      0.159
      short sleeve top        100         42      0.566      0.738       0.65      0.552
            vest dress        100         11      0.208      0.364       0.35      0.292
              trousers        100         32      0.518      0.531      0.565      0.385
       long sleeve top        100         20      0.358        0.7      0.386      0.223
                shorts        100          9      0.419      0.222      0.294      0.198
                 skirt        100         25      0.519       0.64      0.542      0.388
     long sleeve dress        100          8      0.478       0.25       0.27      0.238
   long sleeve outwear        100          5      0.358        0.2      0.369      0.235
Speed: 0.4ms pre-process, 3849.8ms inference, 6.7ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/val/exp17



openvino- int8

val: Scanning /content/drive/MyDrive/yolov9Training_matrice/yolov9/data/test/labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100% 100/100 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 100/100 [03:56<00:00,  2.37s/it]
                   all        100        168      0.365      0.378      0.366      0.257
           sling dress        100          3          0          0      0.206      0.163
                  vest        100          5      0.491      0.196      0.175     0.0907
    short sleeve dress        100          8      0.214      0.309      0.191       0.15
      short sleeve top        100         42      0.503      0.738       0.65      0.535
            vest dress        100         11      0.182      0.364       0.36      0.272
              trousers        100         32      0.489      0.539      0.556      0.357
       long sleeve top        100         20      0.371        0.7      0.396      0.219
                shorts        100          9      0.404      0.222      0.307      0.199
                 skirt        100         25      0.526       0.64      0.541      0.374
     long sleeve dress        100          8      0.475       0.25      0.277      0.237
   long sleeve outwear        100          5      0.358        0.2      0.368      0.227
Speed: 0.4ms pre-process, 2334.0ms inference, 6.7ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/val/exp18



