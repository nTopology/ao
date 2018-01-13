;; Bilinsky dodecahedron
;; en.wikipedia.org/wiki/Bilinski_dodecahedron

(lambda-shape (x y z) (max
  (+ x y -4)
  (+ y z -4)
  (+ x z -4)
  (+ (- x) (- y) -4)
  (+ (- x) (- z) -4)
  (+ (- z) (- y) -4)
  (+ x (- y) -4)
  (+ z (- y) -4)
  (+ (- z) y -4)
  (+ x (- z) -4)
  (+ (- x) z -4)
  (+ (- x) y -4)
))