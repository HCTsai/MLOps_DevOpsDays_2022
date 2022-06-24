
---
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt



---

(pymysql.err.OperationalError) (1267, "Illegal mix of collations (latin1_swedish_ci,IMPLICIT) and (utf8mb4_general_ci,COERCIBLE) for operation '='")

SET collation_connection = 'utf8_general_ci';