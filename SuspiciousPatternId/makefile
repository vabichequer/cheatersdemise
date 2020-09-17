# time window (1, 2, 5, 10 minutes), new dataset?, rewrite dumps?, verbose?, time limitation?, minimum exercises (10, 20)
# 1 -> yes, -> no

new:
	python3 MainCD.py 1 1 0 0 0 10
	python3 MainCD.py 2 1 0 0 0 10
	python3 MainCD.py 5 1 0 0 0 10
	python3 MainCD.py 10 1 0 0 0 10

old:
	python3 MainCD.py 1 0 1 0 0 10
	python3 MainCD.py 2 0 1 0 0 10
	python3 MainCD.py 5 0 1 0 0 10
	python3 MainCD.py 10 0 1 0 0 10

server:
	sbatch -A ada2_serv -p cccmd serverScript1.sh
	sbatch -A ada2_serv -p cccmd serverScript2.sh
	sbatch -A ada2_serv -p cccmd serverScript5.sh
	sbatch -A ada2_serv -p cccmd serverScript10.sh