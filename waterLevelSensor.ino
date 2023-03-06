
#define sensorOut 8
#define sensorPin A1

int waterLevel = 0;
int x;

void setup() {

  pinMode(sensorOut, OUTPUT);
  
  digitalWrite(sensorOut, LOW);
  
  Serial.begin(9600);
}

void loop() {
  int level = readWaterLevel();
  Serial.println(level); 
  delay(100);
}

//This is a function used to get the reading
int readWaterLevel() {
  digitalWrite(sensorOut, HIGH);  
  delay(10);           
  waterLevel = analogRead(A1);
  digitalWrite(sensorOut, LOW);   
  return waterLevel;            
}
