'''Helper module to determine if a point lies within a polygon

Script is based on the following references -

Reference_1_.

.. _Reference_1: https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not

Reference_2_.

.. _Reference_2: https://algorithmtutor.com/Computational-Geometry/Check-if-a-point-is-inside-a-polygon/
'''


class Point:
    '''Point class to define a point'''

    def __init__(self, s1, s2):
        '''Constructor method

        :param s1: s1 coordinate of point
        :type s1: float
        :param s2: s2 coordinate of point
        :type s2: float
        '''

        self.s1 = s1
        self.s2 = s2

        return


def is_within_polygon(polygon, point):
    '''Determine if a point lies within the polygon

    :param polygon: polygon definition using a set of points
    :type polygon: list of points
    :param point: a single point
    :type client: class:'Point'

    :return: True/False: Depending on if point lies within polygon
    :rtype: Bool
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
