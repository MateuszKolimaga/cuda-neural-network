//#include "shallow.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>

#define RELU(a) ((a > 0) ? a : 0)
#define KERNEL_SIZE(n_C) ((n_C * 2 > 8) ? 8 : n_C * 2)


struct layer_param {
	int M;										 //Ilosc tablic podawanych na wejscie
	int pad;									 //Grubosc warstwy zer na krawedziach (zero-padding)
	int F;									 	 //Rozmiar 2D filtra (F x F) 
	int F_MP;									 //Rozmiar 2D filtra do max poolingu (F_MP x F_MP)
	int stride;								 	 //Ilosc przeskakiwanych pikseli przy konwolucji na inkrementacje
	int stride_MP;									 //To samo, tylko przy max poolingu

	int n_Hprev;				  			  		 //Wysokosc tablicy wejsciowej podawanej na wejscie sieci
	int n_Wprev;			      			 		  	 //Szerokosc tablicy wejsciowej podawanej na wejscie sieci
	int n_Cprev;								     	 //Glebokosc tablicy wejsciowej, jednoczesnie musi to byc glebokosc filtra (F x F x C)

	int n_H;	  			  					 //Wysokosc tablicy uzyskanej po konwolucji kernela z wejsciem
	int n_W;
	int n_C;									 //Ilosc filtrow, jednoczesnie glebokosc wyjscia warstwy						

	int n_Hout; 		  							 //Wysokosc tablicy wyjsciowej warstwy 
	int n_Wout;

	double alpha;                                       				 //Predkosc uczenia
};

struct cache_data {
	double** IN;									//Tablica wejsciowa
	double** Z;									//Wynik splotu
	double** A;									//Wynik Aktywacji
	double** OUT;					 				//Poprzedni wynik po max poolingu, jednoczescie wyjscie warstwy sieci
	double** kernel;				 			        //Filtr
	double** dW;									//Gradient kosztu wzgl�dem kerneli
	double** dA;									//Gradient kosztu wzgl�dem warstwy
	double** dAprev;				    				//Gradient kosztu wzgl�dem wyj�cia warstwy n_l - 1
	double** dZ;					    				//Gradient kosztu wzgl�dem wyniku konwolucji
};


void set_random_IN(layer_param, double**&);						//Ustawia losowe wejscie (do testowania)
void set_new_IN(double**&, double**&, layer_param l);
void show_results(layer_param, cache_data&);	//Wyswietla zawartosc koncowych i posrednich wynikow w warstwie
void brief_inf(layer_param, double**);						//Krotka informacja o wyjsciu sieci
void forw_prop(layer_param, cache_data&);      //Najwazniejsza funkcja (konwolucja, aktywacja, maxpooling)
void simple_del(double**&, int);						        //Usuwanie pamieci
void update_param(layer_param&, layer_param&);                  			//Ustawianie nowych parametr�w warstwy
void prep_new_arrays(layer_param, cache_data&);      //Tworzenie nowych tablic wynikowych
void prep_gradients(layer_param, cache_data&);       //Tworzenie gradientow (narazie losowo, bez funkcji kosztu)
void show_gradients(layer_param, cache_data&);
void back_prop(layer_param, cache_data&);


int main() {
	srand(time(NULL));

	int number_of_layers = 2;
	layer_param* l = new layer_param[number_of_layers];
	cache_data* cache = new cache_data[number_of_layers];

	int n_l = 0;
	layer_param l_prev;

	l[n_l].M = 1;
	l[n_l].pad = 0;
	l[n_l].F = 3;
	l[n_l].F_MP = 2;
	l[n_l].stride = 1;
	l[n_l].stride_MP = 2;
	l[n_l].alpha = 0.1;

	int IN_size = 16;								 //Rzeczywisty rozmiar wejscia
	int IN_depth = 1;								 //Rzeczywista glebokosc wejscia

	l[n_l].n_Hprev = IN_size + 2 * l[n_l].pad;
	l[n_l].n_Wprev = IN_size + 2 * l[n_l].pad;
	l[n_l].n_Cprev = IN_depth;

	l[n_l].n_H = int((l[n_l].n_Hprev - l[n_l].F) / l[n_l].stride) + 1;
	l[n_l].n_W = int((l[n_l].n_Wprev - l[n_l].F) / l[n_l].stride) + 1;
	l[n_l].n_C = 1;

	l[n_l].n_Hout = int((l[n_l].n_H - l[n_l].F_MP) / l[n_l].stride_MP) + 1;
	l[n_l].n_Wout = int((l[n_l].n_W - l[n_l].F_MP) / l[n_l].stride_MP) + 1;


	for (n_l = 0; n_l < number_of_layers; n_l++) {
		std::cout << "\n\n#### WARSTWA: " << n_l + 1 << "#### \n";

		if (n_l == 0) set_random_IN(l[n_l], cache[n_l].IN);
		else {
			l_prev = l[n_l - 1];

			update_param(l_prev, l[n_l]);

			set_new_IN(cache[n_l].IN, cache[n_l - 1].OUT, l[n_l]);
		}

		prep_new_arrays(l[n_l], cache[n_l]);

		forw_prop(l[n_l], cache[n_l]);

		prep_gradients(l[n_l], cache[n_l]);

		if (l[n_l].n_H < 25) show_results(l[n_l], cache[n_l]);
		else brief_inf(l[n_l], cache[n_l].OUT);

		//back_prop(l[n_l], cache[n_l]);

		//show_gradients(l[n_l], cache[n_l]);   //Funkcja wyswietla gradient

	}

	return 0;
}

void set_random_IN(layer_param l, double**& IN) {
	IN = new double* [l.M];

	for (int i = 0; i < l.M; i++) {
		IN[i] = new double[l.n_Cprev * l.n_Hprev * l.n_Wprev];
	}

	for (int m = 0; m < l.M; m++) { //Dla kazdego badanego przypadku (np. pojedynczej mapy bajtowej- spektogram)
		for (int h = 0; h < l.n_Hprev; h++) {	//Przejdz po kazdym wierszu 
			for (int w = 0; w < l.n_Wprev; w++) {	 //Przejdz po kazdej kolumnie
				for (int c = 0; c < l.n_Cprev; c++) {	//Przejdz po kazdym kanale (np. dla wejscia w postaci zdjecia rgb - 3 kanaly)
					if (h < l.pad || h > l.n_Hprev - l.pad - 1) IN[m][w + l.n_Wprev * (h + l.n_Hprev * c)] = 0; //Ustawianie zer dla zero paddingu
					else if (w < l.pad || w > l.n_Wprev - l.pad - 1) IN[m][w + l.n_Wprev * (h + l.n_Hprev * c)] = 0;
					else  IN[m][w + l.n_Wprev * (h + l.n_Hprev * c)] = (rand() % 10 + 1)/10.;	//W tablicy wejsciowej beda same wartosci int
				}
			}
		}
	}
}

void set_new_IN(double**& IN, double**& OUT, layer_param l) {
	IN = new double* [l.M];
	for (int i = 0; i < l.M; i++) {
		IN[i] = new double[l.n_Cprev * l.n_Hprev * l.n_Wprev];
	}

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_Cprev; c++) {
			for (int h = 0; h < l.n_Hprev; h++) {
				for (int w = 0; w < l.n_Wprev; w++) {
					IN[m][w + l.n_Wprev * (h + l.n_Hprev * c)] = OUT[m][w + l.n_Wprev * (h + l.n_Hprev * c)];
				}
			}
		}
	}
}

void update_param(layer_param& l_prev, layer_param& l) {
	l.M = l_prev.M;
	l.pad = l_prev.pad;
	l.F = l_prev.F;
	l.F_MP = l_prev.F_MP;
	l.stride = l_prev.stride;
	l.stride_MP = l_prev.stride_MP;
	l.alpha = l_prev.alpha;

	l.n_Hprev = l_prev.n_Hout;
	l.n_Wprev = l_prev.n_Wout;
	l.n_Cprev = l_prev.n_C;

	l.n_H = int((l.n_Hprev - l.F) / l.stride) + 1;
	l.n_W = int((l.n_Wprev - l.F) / l.stride) + 1;
	l.n_C = KERNEL_SIZE(l_prev.n_C);

	l.n_Hout = int((l.n_H - l.F_MP) / l.stride_MP) + 1;
	l.n_Wout = int((l.n_W - l.F_MP) / l.stride_MP) + 1;
}

void prep_new_arrays(layer_param l, cache_data& cache) {
	cache.Z = new double* [l.M];
	cache.A = new double* [l.M];
	cache.OUT = new double* [l.M];
	cache.kernel = new double* [l.n_C];

	for (int i = 0; i < l.M; i++) {
		cache.Z[i] = new double[l.n_C * l.n_H * l.n_W];
		cache.A[i] = new double[l.n_C * l.n_H * l.n_W];
		cache.OUT[i] = new double[l.n_C * l.n_Hout * l.n_Wout];
	}

	for (int i = 0; i < l.n_C; i++) {
		cache.kernel[i] = new double[l.n_Cprev * l.F * l.F];
	}

	for (int c = 0; c < l.n_C; c++) {
		for (int h = 0; h < l.F; h++) {
			for (int w = 0; w < l.F; w++) {
				for (int d = 0; d < l.n_Cprev; d++) {
					cache.kernel[c][w + l.F * (h + l.F * d)] =  (rand()%10000 - 5000) * 0.0001; //Ustawianie losowych wag filtra
				}
			}
		}
	}
}

void prep_gradients(layer_param l, cache_data& cache) {
	cache.dZ = new double* [l.M];
	cache.dA = new double* [l.M];
	cache.dAprev = new double* [l.M];
	cache.dW = new double* [l.n_C];

	for (int i = 0; i < l.M; i++) {
		cache.dZ[i] = new double[l.n_C * l.n_H * l.n_W];
	}
	for (int i = 0; i < l.M; i++) {
		cache.dA[i] = new double[l.n_C * l.n_Hout * l.n_Wout];
	}
	for (int i = 0; i < l.M; i++) {
		cache.dAprev[i] = new double[l.n_Cprev * l.n_Hprev * l.n_Wprev];
	}
	for (int i = 0; i < l.n_C; i++) {
		cache.dW[i] = new double[l.n_Cprev * l.F * l.F];
	}

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			for (int h = 0; h < l.n_Hout; h++) {
				for (int w = 0; w < l.n_Wout; w++) {
					cache.dA[m][w + l.n_Wout * (h + l.n_Hout * c)] = (rand()%10000 - 5000) * 0.0001;
				}
			}
		}
	}

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_Cprev; c++) {
			for (int h = 0; h < l.n_Hprev; h++) {
				for (int w = 0; w < l.n_Wprev; w++) {
					cache.dAprev[m][w + l.n_Wprev * (h + l.n_Hprev * c)] = 0;
				}
			}
		}
	}

	for (int c = 0; c < l.n_C; c++) {
		for (int h = 0; h < l.F; h++) {
			for (int w = 0; w < l.F; w++) {
				for (int d = 0; d < l.n_Cprev; d++) {
					cache.dW[c][w + l.F * (h + l.F * d)] = 0;
				}
			}
		}
	}


	double maximum;
	int vert_start, vert_end;
	int horiz_start, horiz_end;
	for (int m = 0; m < l.M; m++) { //Dla kazdego przypadku
		for (int h = 0; h < l.n_Hout; h++) { //Dla kazdego wiersza wyjscia (wyniku max poolingu)
			for (int w = 0; w < l.n_Wout; w++) { // Dla kazdej kolumny wyjscia
				for (int c = 0; c < l.n_C; c++) { //Dla kazdego kanalu wyjscia

					vert_start = h * l.stride_MP;
					vert_end = vert_start + l.F_MP;
					horiz_start = w * l.stride_MP;
					horiz_end = horiz_start + l.F_MP;

					maximum = 0;

					for (int j = vert_start; j < vert_end; j++) { //Dla kazdego wiersza wycinka wyniku aktywacji
						for (int k = horiz_start; k < horiz_end; k++) { //Dla kazdej kolumny wycinka wyniku aktywacji						
							if (cache.A[m][k + l.n_W * (j + l.n_H * c)] > maximum) maximum = cache.A[m][k + l.n_W * (j + l.n_H * c)];
						}
					}


					for (int j = vert_start; j < vert_end; j++) {
						for (int k = horiz_start; k < horiz_end; k++) {
							if (cache.A[m][k + l.n_W * (j + l.n_H * c)] != maximum || maximum == 0) cache.dZ[m][k + l.n_W * (j + l.n_H * c)] = 0;
							else  cache.dZ[m][k + l.n_W * (j + l.n_H * c)] = cache.dA[m][w + l.n_Wout * (h + l.n_Hout * c)];

						}
					}


				}

			}
		}
	}

	for (int m = 0; m < l.M; m++) {
		for (int h = 0; h < l.n_H; h++) {
			for (int w = 0; w < l.n_W; w++) {
				for (int c = 0; c < l.n_C; c++) {
					vert_start = h;
					vert_end = vert_start + l.F;
					horiz_start = w;
					horiz_end = horiz_start + l.F;

					for (int d = 0; d < l.n_Cprev; d++) {
						for (int j = vert_start; j < vert_end; j++) {
							for (int k = horiz_start; k < horiz_end; k++) {
								if (cache.dZ[m][w + l.n_W * (h + l.n_H * c)] < 0) cache.dZ[m][w + l.n_W * (h + l.n_H * c)] = 0;
								cache.dAprev[m][j + l.n_Wprev * (k + l.n_Hprev * d)] += cache.kernel[c][(k - horiz_start) + l.F * ((j - vert_start) + l.F * d)] *
									cache.dZ[m][w + l.n_W * (h + l.n_H * c)];
								cache.dW[c][(k - horiz_start) + l.F * ((j - vert_start) + l.F * d)] += cache.IN[m][j + l.n_Wprev * (k + l.n_Hprev * d)] *
									cache.dZ[m][w + l.n_W * (h + l.n_H * c)];
							}
						}
					}
				}
			}
		}
	}
}


void brief_inf(layer_param l, double** OUT) {
	for (int m = 0; m < l.M; m++)
		std::cout << "Wyjscie: " << m + 1 << " Kanaly: " << l.n_C << " (" << l.n_Hout << "x" << l.n_Wout << "x" << l.n_C << ")" << "\n" << std::fixed;

}


void forw_prop(layer_param l, cache_data& cache) {
	int M = l.M;
	int pad = l.pad;
	int F = l.F;
	int F_MP = l.F_MP;
	int stride = l.stride;
	int stride_MP = l.stride_MP;

	int n_Hprev = l.n_Hprev;
	int n_Wprev = l.n_Wprev;
	int n_Cprev = l.n_Cprev;

	int n_H = l.n_H;
	int n_W = l.n_W;
	int n_C = l.n_C;

	int n_Hout = l.n_Hout;
	int n_Wout = l.n_Wout;

	int vert_start = 0;
	int vert_end = 0;
	int horiz_start = 0;
	int horiz_end = 0;

	for (int m = 0; m < M; m++) { //Dla kazdego przypadku
		for (int h = 0; h < n_H; h++) {	//Dla kazdego wiersza
			for (int w = 0; w < n_W; w++) { //Dla kazdej kolumny
				for (int c = 0; c < n_C; c++) { //Dla kazdego kanalu (kanalow bedzie tyle, ile chcemy kerneli)
					vert_start = h * stride;     //Poczatek wycinka w pionie
					vert_end = vert_start + F;	 //Koniec wycinka w pionie
					horiz_start = w * stride;	 //Poczatek wycika w poziomie
					horiz_end = horiz_start + F; //Koniec wycinka w poziomie

					cache.Z[m][w + n_W * (h + n_H * c)] = 0;
					for (int d = 0; d < n_Cprev; d++) { //Dla kazdego kanalu w tablicy wejsciowej 
						for (int j = vert_start; j < vert_end; j++) { //Dla wybranych wierszy
							for (int k = horiz_start; k < horiz_end; k++) { //Dla wybranych kolumn
								cache.Z[m][w + n_W * (h + n_H * c)] += cache.kernel[c][(k - horiz_start) + F * ((j - vert_start) + F * d)] *
									cache.IN[m][k + n_Wprev * (j + n_Hprev * d)];
								//Pomnoz wartosc/piksel wycinka przez wage kernela i dodaj do wyniku konwolucji
							}
						}
					}

					cache.A[m][w + n_W * (h + n_H * c)] = RELU(cache.Z[m][w + n_W * (h + n_H * c)]); //Aktywowanie danej wartosci/neuronu

				}
			}
		}
	}

	double maximum = 0;

	for (int m = 0; m < M; m++) { //Dla kazdego przypadku
		for (int h = 0; h < n_Hout; h++) { //Dla kazdego wiersza wyjscia (wyniku max poolingu)
			for (int w = 0; w < n_Wout; w++) { // Dla kazdej kolumny wyjscia
				for (int c = 0; c < n_C; c++) { //Dla kazdego kanalu wyjscia
					if (n_Hout > 1) {
						vert_start = h * stride_MP;
						vert_end = vert_start + F_MP;
						horiz_start = w * stride_MP;
						horiz_end = horiz_start + F_MP;

						maximum = 0;

						for (int j = vert_start; j < vert_end; j++) { //Dla kazdego wiersza wycinka wyniku aktywacji
							for (int k = horiz_start; k < horiz_end; k++) { //Dla kazdej kolumny wycinka wyniku aktywacji
								if (cache.A[m][k + n_W * (j + n_H * c)] > maximum) maximum = cache.A[m][k + n_W * (j + n_H * c)]; //Wybierz maksimum z wycinka
							}
						}

						cache.OUT[m][w + n_Wout * (h + n_Hout * c)] = maximum;
					}
					else
						cache.OUT[m][w + n_Wout * (h + n_Hout * c)] = cache.A[m][0 + n_W * (0 + n_H * c)];

				}

			}
		}
	}
}

void show_gradients(layer_param l, cache_data& cache) {


	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_Cprev; c++) {
			std::cout << "dAprev: " << m + 1 << " Kanal: " << c + 1 << "  (" << l.n_Hprev << "x" << l.n_Wprev << "x" << l.n_Cprev << ")" << "\n";
			for (int h = 0; h < l.n_Hprev; h++) {
				for (int w = 0; w < l.n_Wprev; w++) {
					std::cout << cache.dAprev[m][w + l.n_Wprev * (h + l.n_Hprev * c)] << " ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "\n\n";
	}

	std::cout << "#### dW #### \n\n";

	for (int c = 0; c < l.n_C; c++) {
		for (int d = 0; d < l.n_Cprev; d++) {
			std::cout << "dW: " << c + 1 << " Kanal: " << d + 1 << " (" << l.F << "x" << l.F << "x" << l.n_Cprev << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.F; h++) {
				for (int w = 0; w < l.F; w++) {
					std::cout << cache.dW[c][w + l.F * (h + l.F * d)] << " ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "\n\n";
	}

	std::cout << "#### dZ #### \n\n";

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			std::cout << "dZ: " << m + 1 << " Kanal: " << c + 1 << " (" << l.n_H << "x" << l.n_W << "x" << l.n_C << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.n_H; h++) {
				for (int w = 0; w < l.n_W; w++) {
					std::cout << cache.dZ[m][w + l.n_W * (h + l.n_H * c)] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	std::cout << "#### dA #### \n\n";

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			std::cout << "dA: " << m + 1 << " Kanal: " << c + 1 << " (" << l.n_Hout << "x" << l.n_Wout << "x" << l.n_C << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.n_Hout; h++) {
				for (int w = 0; w < l.n_Wout; w++) {
					std::cout << cache.dA[m][w + l.n_Wout * (h + l.n_Hout * c)] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}
}

void simple_del(double**& arr, int len) {
	for (int i = 0; i < len; i++) {
		delete[] arr[i];
	}
	delete[] arr;
}

void show_results(layer_param l, cache_data& cache)
{
	std::cout.precision(4);

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_Cprev; c++) {
			std::cout << "Wejscie: " << m + 1 << " Kanal: " << c + 1 << "  (" << l.n_Hprev << "x" << l.n_Wprev << "x" << l.n_Cprev << ")" << "\n";
			for (int h = 0; h < l.n_Hprev; h++) {
				for (int w = 0; w < l.n_Wprev; w++) {
					std::cout << cache.IN[m][w + l.n_Wprev * (h + l.n_Hprev * c)] << " ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "\n\n";
	}

	std::cout << "#### FILTRY #### \n\n";

	for (int c = 0; c < l.n_C; c++) {
		for (int d = 0; d < l.n_Cprev; d++) {
			std::cout << "Kernel: " << c + 1 << " Kanal: " << d + 1 << " (" << l.F << "x" << l.F << "x" << l.n_Cprev << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.F; h++) {
				for (int w = 0; w < l.F; w++) {
					std::cout << cache.kernel[c][w + l.F * (h + l.F * d)] << " ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "\n\n";
	}

	std::cout << "#### WYNIKI KONWOLUCJI #### \n\n";

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			std::cout << "Z: " << m + 1 << " Kanal: " << c + 1 << " (" << l.n_H << "x" << l.n_W << "x" << l.n_C << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.n_H; h++) {
				for (int w = 0; w < l.n_W; w++) {
					std::cout << cache.Z[m][w + l.n_W * (h + l.n_H * c)] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	std::cout << "#### WYNIKI AKTYWACJI (RELU) #### \n\n";

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			std::cout << "A: " << m + 1 << " Kanal: " << c + 1 << " (" << l.n_H << "x" << l.n_W << "x" << l.n_C << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.n_H; h++) {
				for (int w = 0; w < l.n_W; w++) {
					std::cout << cache.A[m][w + l.n_W * (h + l.n_H * c)] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	std::cout << "#### WYNIKI MAX POOLINGU #### \n\n";

	for (int m = 0; m < l.M; m++) {
		for (int c = 0; c < l.n_C; c++) {
			std::cout << "Wyjscie: " << m + 1 << " Kanal: " << c + 1 << " (" << l.n_Hout << "x" << l.n_Wout << "x" << l.n_C << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.n_Hout; h++) {
				for (int w = 0; w < l.n_Wout; w++) {
					std::cout << cache.OUT[m][w + l.n_Wout * (h + l.n_Hout * c)] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}
}

void back_prop(layer_param l, cache_data& cache) {

	std::cout << "#### FILTRY PO PO PROPAGACJI WSTECZNEJ #### \n\n";

	for (int c = 0; c < l.n_C; c++) {
		for (int d = 0; d < l.n_Cprev; d++) {
			std::cout << "Wagi: " << c + 1 << " Kanal: " << d + 1 << " (" << l.F << "x" << l.F << "x" << l.n_Cprev << ")" << "\n" << std::fixed;
			for (int h = 0; h < l.F; h++) {
				for (int w = 0; w < l.F; w++) {
					std::cout << cache.kernel[c][w + l.F * (h + l.F * d)] - l.alpha * cache.dW[c][w + l.F * (h + l.F * d)] << " ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "\n\n";
	}

}
