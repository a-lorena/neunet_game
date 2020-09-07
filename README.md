Projekt se sastoji od videoigre i neuronske mreže.

U [requirements.txt](https://github.com/l0renaa/neunet_game/blob/master/requirements.txt) datoteci se nalazi popis svega što je potrebno instalirati za pokretanje igre, navedeni su i brojevi verzija kojima je projekt izrađen.

Može se pokrenuti u tri različita *mode*-a:
  - python neunet.py train --> pokreće treniranje neuronske mreže
  - python neunet.py test --> prikazuje rad već istrenirane neuronske mreže
  - python neunet.py play --> pokreće igru koju korisnik može samostalno igrati
  
U train *mode*-u se stvara direktorij u koji će se spremati svaka 200-ta iteracija, a test *mode* učitava posljednju spremljenu iteraciju.

Cilj igre je prijeći sa lijevog kraja nivoa na desni, sakupljati hranu, izbjegavati neprijatelje i provalije, te na kraju pokupiti zastavicu koja donosi najviše bodova i kraj igre. Različita hrana donosi različiti broj bodova, svaki sudar sa neprijateljem odnosi jedan život, kada se svi životi izgube (pri trećem sudaru) ili upadne u provaliju igra je izgubljena, te se resetira. Sakupljanje zastavice donosi najviše bodova, ispis poruke pobjede i nakon 2 sekunde igra kreće iznova.

Većina datoteka je zajednička neuronskoj mreži i samoj igri, poput enemy.py koja učitava slike neprijatelja, te ih pomiče i animira. Datoteke koje u nazivu sadrže "_for_ai" su namijenje neuronskoj mreži jer su pojednostavljene verzije originalnih datoteka ili učitavaju drugačije podatke. Primjerice player_for_ai.py ne sadrži provjeru pritisnute tipke, ne učitava sve slike, te ne izvodi animacije.

