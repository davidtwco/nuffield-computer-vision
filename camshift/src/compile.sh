/usr/local/angstrom/arm/bin/arm-angstrom-linux-gnueabi-g++ -o compiled/camshift_cross main.cpp -I/usr/local/angstrom/arm/arm-angstrom-linux-gnueabi/usr/include -L/usr/local/angstrom/arm/arm-angstrom-linux-gnueabi/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc

scp compiled/camshift_cross root@192.168.137.121:/home/root/

