CC := /usr/local/angstrom/arm/bin/arm-angstrom-linux-gnueabi-g++
CFLAGS := -O3 -ggdb -mcpu=cortex-a8 -mfloat-abi=softfp -mfpu=neon -ansi -std=gnu99
WARNINGS := -Wall -Wextra -Wno-unused-parameter 
ASSEMBLER := -Wa,

override CFLAGS += $(WARNINGS) $(ASSEMBLER)
LIBS := -lm -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_video -lmathneon 

all: camshift_neon

libmathneon.a: math_atan2f.o math_sqrtf.o math_powf.o math_runfast.o

camshift_neon: camshift.o libmathneon.a
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o:: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

%.a::
	$(AR) rcs $@ $^

clean:
	$(RM) -v camshift_neon *.o *.a
