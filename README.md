# Gravity visualization
## Specifikáció 

Gravitációt demonstráló gumilepedő szimulátor. A lapos tórusz topológiájú (ami kimegy, a szemközti oldalon bejön) gumilepedőnket kezdetben felülről szemléljük, amelyre nagy tömegű, nem mozgó testeket tehetünk a jobb egérgomb lenyomással, és kistömegű golyókat csúsztathatunk súrlódásmentesen a bal alsó sarokból a bal egérgomb lenyomással, amikor a lenyomás helye a bal alsó sarokkal együtt a kezdősebességet adja meg. A nyugalomban lévő nagytömegű testek  görbítik a teret, azaz deformálják a gumilepedőt, de ők nem láthatók. Az okozott benyomódás a tömeg közepétől r távolságra m/(r + r0), ahol r0 a gumilepedő szélességének fél százaléka, m pedig az egymás után felvett testekre egyre növekvő tömeg. A gumilepedő optikailag rücskös, a bemélyedés szerint lépcsőzetesen sötétedő diffúz és ambiens tényezővel. A golyók színes diffúz-spekulárisok, térgörbítő hatásuk és méretük elhanyagolható. SPACE lenyomására a virtuális kameránk az első még nem elnyelt golyóhoz ragad, így az ő szempontját is követhetjük. A tömegekkel ütközött golyók elnyelődnek, a golyók közötti ütközéssel nem kell foglalkozni. A gumilepedőt két pontfényforrás világítja meg, amelyek egymás kezdeti pozíciója körül az alábbi kvaternió szerint forognak (t az idő):

q=[cos(t/4), sin(t/4)cos(t)/2, sin(t/4)sin(t)/2, sin(t/4)√(3/4])

<img src="https://user-images.githubusercontent.com/22593928/147514820-afb9fc55-0ff1-451d-af3e-79fd2034f09b.png" width="500" height="500"/>
<img src="https://user-images.githubusercontent.com/22593928/147514823-5a601207-34e3-4fb4-9cec-104c645c4ac5.png" width="500" height="500"/>
