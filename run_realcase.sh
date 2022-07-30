for i in {10,50,100,150,200,300}
do
    python main.py train $i
    python main.py val $i
done
