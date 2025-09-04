import numpy as np

def radarEquation(rangeVal: int | float,
                  transmitPower: int | float = 1,
                  gain: int | float = 1,
                  effectiveAntennaArea: int | float = 1,
                  sigma: int | float = 1
                  ) -> float:
    """ Basic Radar Equation
    
    # TODO - add reference.

    Parameters
    ----------
    rangeVal : int | float
        Range [m].
    transmitPower : int | float, optional
        Transmittion power [W], by default 1.
    gain : int | float, optional
        Gain, by default 1.
    effectiveAntennaArea : int | float, optional
        Effective antenna area [m], by default 1.
    sigma : int | float, optional
        #TODO, by default 1.

    Returns
    -------
    float
        Received power [W].
    """

    # TODO - check why this had an error 
    return transmitPower * gain * effectiveAntennaArea * sigma / ((4 * np.pi)**2 * float(rangeVal)**4)

def rangeResolution(tc: int | float | None = None,
                    bw: int | float | None = None,
                    c: int | float = 3e8
                    ) -> float:
    """ Calculates the radar range resolution
    
    Calculates the range resolution of a radar system. This is the minimum seperation distance requried to distiguish
    between targets. Calculates this either using the time constant or bandwidth. Needs a value for at least one of 
    these values.

    Parameters
    ----------
    tc : int | float | None, optional
        Time constant of pulse [s], by default None.
    bw : int | float | None, optional
        Bandwidth of pulse [Hz], by default None.
    c : int | float, optional
        Speed of light [m/s], by default 3e8.

    Returns
    -------
    float
        Range resolution [m].

    Raises
    ------
    ValueError
        If time constant of bandwidth are both set to None.
    """
    
    if tc is not None:
        res = c * tc / 2
    elif bw is not None:
        res = c / (2 * bw)
    else:
        raise ValueError("need a value for either 'tc' or 'bw'")
    
    return res

def beatFrequency(r: int | float,
                  bw: int | float,
                  tc: int | float,
                  c: int | float = 3e8
                  ) -> float:
    """ Calculates the beat frequency

    The beat frequency is the difference between the frequency between the tx chip and the rx chirp. The difference 
    occours due to the time delay between the tx and rx chirp signal due when it is reflected back off an object. Only
    used for CW and FMCw radar systems.
    
    Parameters
    ----------
    r : int | float
        Range of target [m].
    bw : int | float
        Bandwidth [Hz].
    tc : int | float
        Time constant of pulse [s].
    c : int | float, optional
        Speed of light [m/s], by default 3e8

    Returns
    -------
    float
        Beat frequency [Hz].
    """
    
    return (2 * r * bw) / (c * tc)

def phaseDifference(fTrans: int | float,
                    velocity: int | float,
                    tc: int | float,
                    c: int | float = 3e8
                    ) -> float:
    """ Calculates the phase difference of a transmitted pulse.

    Parameters
    ----------
    fTrans : int | float
        Transmitter frequency [Hz].
    velocity : int | float
        Target velocity [m/s].
    tc : int | float
        Time constant of pulse [s].
    c : int | float, optional
        Speed of light [m/s], by default 3e8.

    Returns
    -------
    float
        Phase difference [radians].
    """
    
    timeDiff = 2 * velocity * tc / c
    
    return 2 * np.pi * fTrans * timeDiff
    

if __name__ == "__main__":
    
    print("\nRADAR UTILS\n")
    
    # radar equation
    print(f"Received Power: {radarEquation(rangeVal=30000):.4} W")
    
    # range resolution
    print(f"Range Resolution: {rangeResolution(tc=0.1)} m")
    print(f"Range Resolution: {rangeResolution(bw=1.6e9)} m")
    
    # beat frequency
    print(f"Beat frequency {beatFrequency(r=20, bw=1.6e9, tc=40e-6): .2f} Hz")
    
    # phase difference
    print(f"Phase difference {phaseDifference(fTrans=79e9, velocity=10, tc=40e-6): .2f} radians")
    
    print(" ")