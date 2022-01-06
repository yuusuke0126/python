#!/usr/bin/env python3

import thingspeak
import time
import datetime
from remo import NatureRemoAPI

ch_living = thingspeak.Channel(id=815673, api_key='R04RZOTRZIKJN039', fmt='json', timeout=None)  # Living room
ch_bedroom = thingspeak.Channel(id=1610793, api_key='D4JHGVHK0VHL67QI', fmt='json', timeout=None)  # bed room
api = NatureRemoAPI('KxQTPvV_suCMtvDsD7yIwl6bUmPa9AS1yAR7D1_sSdY.O4X2Bxy100mwe2l3Z1BPDjCnP3cqzLw6MxgImm4RIRw')
is_humi = False
strong_mode = False
turn_on = '998cef55-eb23-4c92-8f57-1a55de14138e'
turn_off = 'c2a0c610-8388-401e-9745-e9028dfe04d0'
heat = 'a3cb4191-be43-401e-ab5a-731c89b34448'
uv = '3c7327ca-8cff-40e8-b8d4-39b8bfe45032'
mode_change = '269ec445-7676-4524-aeb3-04a10e8a44bb'

def humi_on():
    api.send_signal(turn_on)
    time.sleep(3)
    api.send_signal(uv)
    time.sleep(3)
    api.send_signal(heat)
    time.sleep(3)
    return True, True

def humi_off():
    api.send_signal(turn_off)
    time.sleep(3)
    return False, False

def humi_mode_change(is_strong):
    if is_strong:
        api.send_signal(mode_change)
        time.sleep(3)
        return False
    else:
        humi_off()
        humi_on()
        return True

while True:
    try:
        devices = api.get_devices()
        temp_living = devices[0].newest_events['te'].val
        humi_living = devices[0].newest_events['hu'].val
        illu_living = devices[0].newest_events['il'].val
        temp_bedroom = devices[1].newest_events['te'].val
        humi_bedroom = devices[1].newest_events['hu'].val
        res0 = ch_living.update({1: temp_living, 2: humi_living, 3: illu_living})
        res1 = ch_bedroom.update({1: temp_bedroom, 2: humi_bedroom})
        if illu_living > 5:
            if is_humi is False:
                if humi_living < 50:
                    is_humi, strong_mode = humi_on()
            else:
                if humi_living > 60:
                    is_humi, strong_mode = humi_off()
                elif humi_living > 57 and strong_mode:
                    strong_mode = humi_mode_change(strong_mode)
                elif humi_living < 53 and strong_mode is False:
                    strong_mode = humi_mode_change(strong_mode)
        elif is_humi:
            is_humi, strong_mode = humi_off()
        time.sleep(600)
    except Exception as e:
        print(str(datetime.datetime.now()) + ": Error occurs...")
        print(e)
        time.sleep(60)

