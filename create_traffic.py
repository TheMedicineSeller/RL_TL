import numpy as np
import math

# Should be run every time before episode

def gen_traffic(seed, n_cars=1000, max_steps=4000):
    np.random.seed(seed)  # make tests reproducible
    # the generation of cars is distributed according to a weibull distribution
    timings = np.random.weibull(2, n_cars)
    timings = np.sort(timings)
    # reshape the distribution to fit the interval 0:max_steps
    
    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = max_steps
    
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)
    car_gen_steps = np.rint(car_gen_steps)
    
    with open("sumo_files/handmade.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" vClass="passenger" length="3" accel="1.0" decel="4.5" sigma="0.5" maxSpeed="20"/>
        
        <!-- straight routes -->
        <route id="N1S1" edges="NTL_1 TLS_1"/>
        <route id="N2S2" edges="NTL_2 TLS_2"/>
        <route id="S1N1" edges="STL_1 TLN_1"/>
        <route id="S2N2" edges="STL_2 TLN_2"/>
        <route id="WE" edges="WTL_1 TL_12 TLE_2"/>
        <route id="EW" edges="ETL_2 TL_21 TLW_1"/>
    
        <!-- turn routes -->
        <route id="ES1" edges="ETL_2 TL_21 TLS_1"/>
        <route id="S1E" edges="STL_1 TL_12 TLE_2"/>
        <route id="ES2" edges="ETL_2 TLS_2"/>
        <route id="S2E" edges="STL_2 TLE_2"/>
        <route id="EN1" edges="ETL_2 TL_21 TLN_1"/>
        <route id="N1E" edges="NTL_1 TL_12 TLE_2"/>
        <route id="EN2" edges="ETL_2 TLN_2"/>
        <route id="N2E" edges="NTL_2 TLE_2"/>
    
        <route id="WS1" edges="WTL_1 TLS_1"/>
        <route id="S1W" edges="STL_1 TLW_1"/>
        <route id="WS2" edges="WTL_1 TL_12 TLS_2"/>
        <route id="S2W" edges="STL_2 TL_21 TLW_1"/>
        <route id="WN1" edges="WTL_1 TLN_1"/>
        <route id="N1W" edges="NTL_1 TLW_1"/>
        <route id="WN2" edges="WTL_1 TL_12 TLN_2"/>
        <route id="N2W" edges="NTL_2 TL_21 TLW_1"/>
    
        <route id="N1S2" edges="NTL_1 TL_12 TLS_2"/>
        <route id="S2N1" edges="STL_2 TL_21 TLN_1"/>
        <route id="N2S1" edges="NTL_2 TL_21 TLS_1"/>
        <route id="S1N2" edges="STL_1 TL_12 TLN_2"/>
        <route id="S1S2" edges="STL_1 TL_12 TLS_2"/>
        <route id="S2S1" edges="STL_2 TL_21 TLS_1"/>
        <route id="N1N2" edges="NTL_1 TL_12 TLN_2"/>
        <route id="N2N1" edges="NTL_2 TL_21 TLN_1"/>
        """, file=routes)
    
        for car_counter, step in enumerate(car_gen_steps):
            straight_or_turn = np.random.uniform()
            if straight_or_turn < 0.5:
                route_straight = np.random.randint(1, 7)  # choose a random source & destination
                if route_straight == 1:
                    print('    <vehicle id="N1S1_%i" type="car" route="N1S1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 2:
                    print('    <vehicle id="N2S2_%i" type="car" route="N2S2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 3:
                    print('    <vehicle id="S1N1_%i" type="car" route="S1N1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 4:
                    print('    <vehicle id="S2N2_%i" type="car" route="S2N2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 5:
                    print('    <vehicle id="WE%i" type="car" route="WE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    print('    <vehicle id="EW%i" type="car" route="EW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                
            else:
                route_turn = np.random.randint(1, 25)  # choose random source & destination
                if route_turn == 1:
                    print('    <vehicle id="ES1_%i" type="car" route="ES1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 2:
                    print('    <vehicle id="S1E_%i" type="car" route="S1E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 3:
                    print('    <vehicle id="ES2_%i" type="car" route="ES2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 4:
                    print('    <vehicle id="S2E_%i" type="car" route="S2E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 5:
                    print('    <vehicle id="EN1_%i" type="car" route="EN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 6:
                    print('    <vehicle id="N1E_%i" type="car" route="N1E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 7:
                    print('    <vehicle id="EN2_%i" type="car" route="EN2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 8:
                    print('    <vehicle id="N2E_%i" type="car" route="N2E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 9:
                    print('    <vehicle id="WS1_%i" type="car" route="WS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 10:
                    print('    <vehicle id="S1W_%i" type="car" route="S1W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 11:
                    print('    <vehicle id="WS2_%i" type="car" route="WS2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 12:
                    print('    <vehicle id="S2W_%i" type="car" route="S2W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 13:
                    print('    <vehicle id="WN1_%i" type="car" route="WN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 14:
                    print('    <vehicle id="N1W_%i" type="car" route="N1W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 15:
                    print('    <vehicle id="WN2_%i" type="car" route="WN2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 16:
                    print('    <vehicle id="N2W_%i" type="car" route="N2W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 17:
                    print('    <vehicle id="N1S2_%i" type="car" route="N1S2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 18:
                    print('    <vehicle id="S2N1_%i" type="car" route="S2N1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 19:
                    print('    <vehicle id="N2S1_%i" type="car" route="N2S1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 20:
                    print('    <vehicle id="S1N2_%i" type="car" route="S1N2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 21:
                    print('    <vehicle id="S1S2_%i" type="car" route="S1S2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 22:
                    print('    <vehicle id="S2S1_%i" type="car" route="S2S1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_turn == 23:
                    print('    <vehicle id="N1N2%i" type="car" route="N1N2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    print('    <vehicle id="N2N1_%i" type="car" route="N2N1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
    
        print("\n</routes>", file=routes)
