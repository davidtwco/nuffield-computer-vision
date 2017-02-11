################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cld/ETF.cpp \
../src/cld/fdog.cpp 

OBJS += \
./src/cld/ETF.o \
./src/cld/fdog.o 

CPP_DEPS += \
./src/cld/ETF.d \
./src/cld/fdog.d 


# Each subdirectory must supply rules for building sources it contributes
src/cld/%.o: ../src/cld/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


