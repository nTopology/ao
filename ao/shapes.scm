(define-module (ao shapes))

(use-modules (ao overload))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; 2D shapes
(define-public (circle center r)
    " Constructs a circle from a center '(x0 y0 z0) and radius "
    (move (lambda (x y z) (- (sqrt (+ (* x x) (* y y))) r))
          center))

(define-public (rect a b)
    " Constructs a rectangle from two '(x y) lists representing corners "
    (let* ((xa (car a))
           (xb (car b))
           (ya (cadr a))
           (yb (cadr b))
           (xmin (min xa xb))
           (xmax (max xa xb))
           (ymin (min ya yb))
           (ymax (max ya yb)))
    (lambda (x y z) (max (- xmin x) (- x xmax)
                         (- ymin y) (- y ymax)))))

(define-public (triangle a b c)
    " Constructs a triangle from three '(x y) lists"
    (let* ;; Find the center point of the triangle
          ((xm (/ (+  (car a)  (car b)  (car c)) 3))
           (ym (/ (+ (cadr a) (cadr b) (cadr c)) 3))

           ;; Calculate the angles of each point about the center
           (get-angle (lambda (p) (atan (- (car p) xm) (- (cadr p) ym))))

           ;; Extract the three angles
           (ta (get-angle a))
           (tb (get-angle b))
           (tc (get-angle c))

           ;; Rotate the points so that the smallest angle is first
           (sorted (cond ((and (< tb ta) (< tb tc)) (list tb tc ta))
                         ((and (< tc ta) (< tc tb)) (list tc ta tb))
                         (else (list ta tb tc))))

           ;; If the points are in increasing order, it's a sign that
           ;; they aren't sorted in a clockwise fashion.  In this case, swap
           ;; the second and third point to make them clockwise
           (clockwise (if (> (caddr sorted) (cadr sorted))
                          (list a c b) (list a b c)))

           ;; Extract coordinates from sorted list
           (x0 (caar clockwise))   (y0 (cdar clockwise))
           (x1 (caadr clockwise))  (y1 (cdadr clockwise))
           (x2 (caaddr clockwise)) (y2 (cdaddr clockwise))

           ;; Function to calculate one edge of the triangle
           (edge (lambda (x0 y0 dx dy x y)
                    (- (* dy (- x x0)) (* dx (- y y0))))))

          (lambda (x y z) (max (edge x2 y2 (- x0 x2) (- y0 y2) x y)
                               (edge x1 y1 (- x2 x1) (- y2 y1) x y)
                               (edge x0 y0 (- x1 x0) (- y1 y0) x y)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; 2D -> 3D functions

(define-public (extrude shape bounds)
    "Extrudes the given shape given bounds '(zmin zmax)"
    (let ((zmin (apply min bounds))
          (zmax (apply max bounds)))
    (lambda (x y z) (max (shape x y z) (- zmin z) (- z zmax)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; 3D shapes

(define-public (sphere center r)
    "Defines a sphere from a center '(x0 y0 z0) and radius r"
    (move (lambda (x y z) (- (sqrt (+ (* x x) (* y y) (* z z))) r))
          center))

(define-public (cube a b)
    "Defines a cube with corners a and b, each a list '(x y z)"
    (extrude (rect a b) (list (caddr a) (caddr b))))

(define-public (cylinder base r height)
    "Creates a circle with the given base '(x y z), radius r, and height dz"
    (extrude (circle base r) (list (caddr base) height)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; CSG operations
(define-public (union . shapes)
    "Returns the union of a set of shapes"
    (lambda (x y z) (apply min (map (lambda (s) (s x y z)) shapes))))

(define-public (intersection . shapes)
    "Returns the intersection of a set of shapes"
    (lambda (x y z) (apply max (map (lambda (s) (s x y z)) shapes))))

(define-public (difference a . b)
    "Subtract a set of shapes from a target shape"
    (let ((accum (apply union b)))
    (lambda (x y z) (max (a x y z)
                         (-(accum x y z))))))

(define-public (offset s o)
    "Expand or contract a given shape s by an offset o"
    (lambda (x y z) (- (s x y z) o)))

(define-public (clearance a b o)
    "Expands shape b by offset o then subtracts it from shape a"
    (let ((bo (offset b o)))
    (lambda (x y z) (max (a x y z) (- (bo x y z))))))

(define-public (shell a o)
    "Returns a shell of shape a offset by amount o"
    (clearance a a o))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Transforms
(define-public (move f delta)
    "Moves a shape by delta '(dx dy [dz=0])"
    (let ((dx   (car delta))
          (dy  (cadr delta))
          (dz (if (>= (length delta) 3) (caddr delta) 0)))
    (lambda (x y z) (f (- x dx) (- y dy) (- z dz)))))

(define-public (rotate-z shape angle . args)
    "Rotate the given shape by an angle in radians
The center of rotation is 0,0 or specified by optional argument '(x0 y0)"
    (let* ((argc (length args))
           (x0 (if (> argc 0) (caar  args) 0))
           (y0 (if (> argc 0) (cadar args) 0))
           (ca (cos angle))
           (sa (sin angle)))
    (lambda (x y z) (shape (+ (* ca x) (* sa y))
                           (+ (* (- sa) x) (* ca y))
                           z))))

(define-public (reflect-x shape . args)
    "Reflect the given shape about the x origin or an optional argument x0"
    (let ((x0 (if (> 0 (length args)) (car args) 0)))
    (lambda (x y z) (shape (- (* 2 x0) x) y z))))

(define-public (reflect-y shape . args)
    "Reflect the given shape about the y origin or an optional argument y0"
    (let ((y0 (if (> 0 (length args)) (car args) 0)))
    (lambda (x y z) (shape x (- (* 2 y0) y) z))))

(define-public (reflect-z shape . args)
    "Reflect the given shape about the z origin or an optional argument z0"
    (let ((z0 (if (> 0 (length args)) (car args) 0)))
    (lambda (x y z) (shape x y (- (* 2 z0) z)))))
