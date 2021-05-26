'''Helper module to determine if a point lies within a polygon

Script is based on Ref1_ and Ref2_.

.. _Ref1: https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not

.. _Ref2: https://algorithmtutor.com/Computational-Geometry/Check-if-a-point-is-inside-a-polygon/
'''


class Point:
    '''Point class to define a point'''

    def __init__(self, s1, s2):
        '''
        Constructor method

        Parameters
        ----------
        s1: float
            s1 coordinate of point
        s2: float
            s2 coordinate of point

        Returns
        -------
        None
        '''

        self.s1 = s1
        self.s2 = s2

        return


def is_within_polygon(polygon, point):
    '''
    Determine if a point lies within the polygon

    Parameters
    ----------
    polygon: list of points
        polygon definition using a set of points
    point: class:'Point'
        a single point

    Returns
    -------
    True/False: Bool
        Depending on if point lies within polygon
    '''

    A = []
    B = []
    C = []

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        # calculate A, B and C
        a = -(p2.s2 - p1.s2)
        b = p2.s1 - p1.s1
        c = -(a * p1.s1 + b * p1.s2)

        A.append(a)
        B.append(b)
        C.append(c)

    D = []
    for i in range(len(A)):
        d = A[i] * point.s1 + B[i] * point.s2 + C[i]
        D.append(d)

    t1 = all(d >= 0 for d in D)
    t2 = all(d <= 0 for d in D)

    return t1 or t2


if __name__ == '__main__':
    # Example usage:
    # define a polygon containing as many points representing vertices
    polygon = [Point(0, 0), Point(5, 0), Point(6, 7),
               Point(2, 3), Point(0, 4)]

    # define a point p1
    p1 = Point(1, 1)
    # determine if p1 lies within the polygon
    print(is_within_polygon(polygon, p1))  # returns True
    p2 = Point(6, 2)
    print(is_within_polygon(polygon, p2))  # returns False
