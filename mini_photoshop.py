import tkinter as tk
from tkinter import filedialog, Menu
from tkinter import PhotoImage

from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt  # Certifique-se de importar a biblioteca


# Função para carregar imagem
def load_image():
    global image, img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))  # Convertendo para RGB para exibir no Tkinter
        image = img_pil
        img_tk = ImageTk.PhotoImage(image)
        original_label.config(image=img_tk)
        original_label.image = img_tk

# Função para aplicar filtros e mostrar resultado
def apply_filter(filter_func):
    if image:
        filtered_img = filter_func()
        img_tk = ImageTk.PhotoImage(filtered_img)
        filtered_label.config(image=img_tk)
        filtered_label.image = img_tk

# Filtro monocromático com OpenCV
def monochrome():
    global img_cv
    if img_cv is not None:
        b, g, r = img_cv[:,:,0], img_cv[:,:,1], img_cv[:,:,2]
        img_gray = 0.299 * r + 0.587 * g + 0.114 * b
        img_gray = img_gray.astype(np.uint8)
        img_pil = Image.fromarray(img_gray)
        return img_pil



def blur():
    global image
    return image.filter(ImageFilter.BLUR)

def binarization():
    global img_cv
    if img_cv is not None:
        # Converter a imagem para escala de cinza usando a função monochrome
        img_gray = monochrome()
        
        # Verificar dimensões da imagem
        largura, altura = img_gray.size  # Corrigido para usar largura e altura corretamente
        
        # Criar uma cópia para binarização
        img_bin = np.zeros((altura, largura), dtype=np.uint8)  # Ajustado para formato correto
        
        # Definir o limiar de binarização
        limiar = 100
        
        # Realizar binarização manualmente
        for i in range(altura):
            for j in range(largura):
                valor = 0 if img_gray.getpixel((j, i)) < limiar else 255  # Corrigido para acessar as coordenadas corretamente
                img_bin[i, j] = valor
        
        # Converter para PIL Image para visualização
        img_pil = Image.fromarray(img_bin)
        return img_pil

def label_connected_components():
    global img_cv
    if img_cv is not None:
        img_bin = binarization()  # Binariza a imagem
        img_bin_np = np.array(img_bin)  # Converte para numpy array
        largura, altura = img_bin_np.shape
        
        # Matriz para armazenar rótulos
        labels = np.zeros((altura, largura), dtype=int)
        current_label = 1  # Inicia o rótulo a partir de 1
        
        # Algoritmo de rotulagem com vizinhança de 4
        for i in range(altura):
            for j in range(largura):
                if img_bin_np[i, j] == 255 and labels[i, j] == 0:  # Se é um objeto e não rotulado
                    # Rotular a nova região
                    labels[i, j] = current_label
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        # Verificar vizinhos (cima, baixo, esquerda, direita)
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Vizinhança 4
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < altura and 0 <= ny < largura:  # Verificar limites
                                if img_bin_np[nx, ny] == 255 and labels[nx, ny] == 0:
                                    labels[nx, ny] = current_label
                                    stack.append((nx, ny))
                    current_label += 1  # Incrementar rótulo para próximo objeto
        
        # Criar imagem colorida para visualização dos rótulos
        label_color_image = np.zeros((altura, largura, 3), dtype=np.uint8)
        for label in range(1, current_label):
            color = np.random.randint(0, 255, 3)  # Cor aleatória
            label_color_image[labels == label] = color
        
        img_pil = Image.fromarray(label_color_image)

        # Atualizar o rótulo com a quantidade de rótulos encontrados
        label_count.config(text=f"Quantidade de Rótulos: {current_label - 1}")
        
        return img_pil

def sharpen():
    global image
    return image.filter(ImageFilter.SHARPEN)

def negative():
    global img_cv
    if img_cv is not None:
        # Converter para escala de cinza
        img_gray = monochrome()

        # Converter a imagem PIL para uma matriz NumPy
        img_gray_np = np.array(img_gray)
        
        largura, altura = img_gray.size
        img_neg = np.zeros((altura, largura), dtype=np.uint8)
        
        # Criar imagem negativa manualmente
        for i in range(altura):
            for j in range(largura):
                img_neg[i, j] = 255 - img_gray_np[i, j]  # Subtração para o negativo
        
        # Converter de volta para imagem PIL
        img_pil = Image.fromarray(img_neg)
        return img_pil

def power_transform(gamma=0.5):
    global img_cv
    if img_cv is not None:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray / 255.0
        img_corrected = np.power(img_normalized, gamma)
        img_corrected = np.uint8(img_corrected * 255)
        img_pil = Image.fromarray(img_corrected)
        return img_pil

def contrast_stretch():
    global img_cv
    if img_cv is not None:
        # Converter a imagem para escala de cinza
        img_gray = monochrome()
        
        # Converter a imagem PIL em uma matriz NumPy
        img_gray_np = np.array(img_gray)

        # Obter os valores mínimos e máximos de intensidade
        min_val = np.min(img_gray_np)
        max_val = np.max(img_gray_np)

        # Criar uma matriz vazia para armazenar a imagem ajustada
        altura, largura = img_gray_np.shape  # Usar a forma da matriz NumPy
        img_stretched = np.zeros((altura, largura), dtype=np.uint8)

        # Aplicar a fórmula de alargamento de contraste manualmente
        for i in range(altura):
            for j in range(largura):
                # Novo valor: ((pixel - min) / (max - min)) * 255
                img_stretched[i, j] = int(((img_gray_np[i, j] - min_val) / (max_val - min_val)) * 255)

        # Converter de volta para uma imagem PIL para exibição no Tkinter
        img_pil = Image.fromarray(img_stretched)
        return img_pil

def adaptive_binarization():
    global img_cv
    if img_cv is not None:
        img_gray = monochrome()
        img_gray_np = np.array(img_gray)

        # Aplicar binarização adaptativa
        # Usamos um bloco de 11x11 pixels com uma constante de 2 para ajustar o limiar localmente
        img_bin = np.zeros_like(img_gray_np)
        altura, largura = img_gray_np.shape

        for i in range(altura):
            for j in range(largura):
                # Calcula a média local de um bloco de 11x11 ao redor de (i,j)
                x_min = max(i - 5, 0)
                x_max = min(i + 5, altura - 1)
                y_min = max(j - 5, 0)
                y_max = min(j + 5, largura - 1)
                
                bloco = img_gray_np[x_min:x_max + 1, y_min:y_max + 1]
                media_local = np.mean(bloco)
                
                # Se o valor do pixel for maior que a média local - 2, torna branco, caso contrário, preto
                img_bin[i, j] = 255 if img_gray_np[i, j] > (media_local - 2) else 0

        img_pil = Image.fromarray(img_bin)
        return img_pil
 

def find_and_draw_rectangles():
    global img_cv
    if img_cv is not None:
        # Passo 1: Converter para monocromático e binarizar usando o novo método
        img_bin = adaptive_binarization()
        img_bin_np = np.array(img_bin)  # Converter para numpy array

        # Passo 2: Encontrar contornos
        altura, largura = img_bin_np.shape
        contours = []
        visited = np.zeros_like(img_bin_np)

        for i in range(1, altura - 1):
            for j in range(1, largura - 1):
                if img_bin_np[i, j] == 255 and visited[i, j] == 0:
                    contour = []
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if visited[x, y] == 0:
                            visited[x, y] = 1
                            contour.append((x, y))
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < altura and 0 <= ny < largura:
                                    if img_bin_np[nx, ny] == 255 and visited[nx, ny] == 0:
                                        stack.append((nx, ny))
                    if contour:
                        contours.append(contour)

        # Passo 3: Filtrar contornos pequenos
        min_contour_size = 50  # Tamanho mínimo do contorno para considerar
        filtered_contours = [c for c in contours if len(c) > min_contour_size]

        # Passo 4: Desenhar retângulos ao redor dos contornos
        img_with_rectangles = cv2.cvtColor(img_bin_np, cv2.COLOR_GRAY2BGR)  # Converter para BGR para manter a imagem original
        for contour in filtered_contours:
            min_x = min([p[1] for p in contour])
            max_x = max([p[1] for p in contour])
            min_y = min([p[0] for p in contour])
            max_y = max([p[0] for p in contour])

            # Desenhar o retângulo com uma borda mais espessa (ex: 3px) e cor rosa
            cv2.rectangle(img_with_rectangles, (min_x, min_y), (max_x, max_y), (255, 0, 255), 3)  # Cor rosa, borda 3px

        img_pil = Image.fromarray(img_with_rectangles)
        return img_pil




def histogram_equalization():
    global img_cv
    if img_cv is not None:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        img_pil = Image.fromarray(img_eq)
        return img_pil


def histogram_data():
    global img_cv
    if img_cv is not None:
        # Obtenha a imagem em escala de cinza
        img_gray = monochrome()
        largura, altura = img_gray.size
        
        # Converta a imagem PIL em uma matriz NumPy
        img_gray_np = np.array(img_gray)
        
        # Criar o vetor para o histograma (0-255)
        x = list(range(256))
        hist = np.zeros(256, dtype=int)  # Use int para o histograma
        
        # Calcular o histograma
        for i in range(altura):
            for j in range(largura):
                pixel = img_gray_np[i, j]
                hist[pixel] += 1
        
        # Exibir o histograma usando matplotlib
        plt.figure(figsize=(6, 4))
        plt.plot(x, hist, color='black')  # Pode mudar a cor se desejar
        plt.xlabel('Intensidades')
        plt.ylabel('Quantidade de Pixels')
        plt.title('Histograma')
        plt.grid(True)
        plt.xlim(0, 255)  # Limite do eixo x
        plt.ylim(0, max(hist) + 10)  # Limite do eixo y
        plt.show()  # Exibir o gráfico

def mean_filter():
    global img_cv
    if img_cv is not None:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.float32) / 9
        img_filtered = cv2.filter2D(img_gray, -1, kernel)
        img_pil = Image.fromarray(img_filtered)
        return img_pil

def log_transform():
    global img_cv
    if img_cv is not None:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        c = 244 / np.log(1 + np.max(img_gray))
        img_log = c * np.log(1 + img_gray)
        img_log = np.uint8(img_log)
        img_pil = Image.fromarray(img_log)
        return img_pil

# Função para sair do aplicativo
def exit_app():
    root.quit()

# Inicializar janela

root = tk.Tk()
root.title("Mini Photoshop")
root.iconbitmap('chave.ico')

# Maximizar a janela na tela do usuário
root.state('zoomed')

# Criar barra de menus
menu_bar = Menu(root)

file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Abrir Imagem", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Sair", command=exit_app)
menu_bar.add_cascade(label="Arquivo", menu=file_menu)

root.config(menu=menu_bar)

left_frame = tk.Frame(root, width=200, bg="#2c2c2c")
left_frame.pack(side="left", fill="y")

right_frame = tk.Frame(root, bg="#333333")
right_frame.pack(side="right", fill="both", expand=True)

original_label = tk.Label(right_frame, bg="#333333")
original_label.pack(side="left", padx=10, pady=10)

filtered_label = tk.Label(right_frame, bg="#333333")
filtered_label.pack(side="right", padx=10, pady=10)
label_count = tk.Label(left_frame, text="Quantidade de Rótulos: 0", bg="#2c2c2c", fg="white")
label_count.pack(pady=10)
# Função para estilizar botões com ícones
def create_icon_button(parent, icon_path, command, tooltip):
    img = Image.open(icon_path)
    img = img.resize((40, 40), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    
    button = tk.Button(
        parent, 
        image=photo, 
        command=command, 
        bg="#444444",        # Cor de fundo do botão
        bd=0,                # Sem borda padrão
        relief="flat",       # Estilo do botão sem relevo
        padx=10,             # Padding interno horizontal
        pady=10              # Padding interno vertical
    )
    
    button.image = photo  # Manter uma referência ao ícone
    
    # Ajustar espaço interno (padding) e preencher o espaço horizontal
    button.pack(pady=5, fill="x")

    # Tooltip (texto que aparece ao passar o mouse)
    button.bind("<Enter>", lambda e: show_tooltip(e, tooltip))
    button.bind("<Leave>", hide_tooltip)
    
    return button
# Funções para tooltip
tooltip_window = None

def show_tooltip(event, tooltip_text):
    global tooltip_window
    tooltip_window = tk.Toplevel(root)
    tooltip_window.wm_overrideredirect(True)
    tooltip_window.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
    label = tk.Label(tooltip_window, text=tooltip_text, background="white", relief="solid", borderwidth=1)
    label.pack()

def hide_tooltip(event):
    global tooltip_window
    if tooltip_window:
        tooltip_window.destroy()
        tooltip_window = None

# Botões com ícones
create_icon_button(left_frame, 'blur.png', lambda: apply_filter(blur), "Desfoque")
create_icon_button(left_frame, 'sharpen.png', lambda: apply_filter(sharpen), "Nitidez")
create_icon_button(left_frame, 'monochrome.png', lambda: apply_filter(monochrome), "Monocromático")
create_icon_button(left_frame, 'negative.png', lambda: apply_filter(negative), "Negativo")
create_icon_button(left_frame, 'retangulo.png', lambda: apply_filter(find_and_draw_rectangles), "Detecção de Retângulos")

create_icon_button(left_frame, 'binario.png', lambda: apply_filter(binarization), "Binarização")
create_icon_button(left_frame, 'logaritmo.png', lambda: apply_filter(log_transform), "Transformada Logarítmica")
create_icon_button(left_frame, 'luz.png', lambda: apply_filter(lambda: power_transform(0.5)), "Transformação de Potência (Gama = 0.5)")
create_icon_button(left_frame, 'ajuste.png', lambda: apply_filter(contrast_stretch), "Alargamento de Contraste")
create_icon_button(left_frame, 'histograma.png', lambda: apply_filter(histogram_equalization), "Equalização de Histograma")
create_icon_button(left_frame, 'medir.png', lambda: apply_filter(mean_filter), "Filtro da Média")
create_icon_button(left_frame, 'grafico.png', lambda: apply_filter(histogram_data), "Histograma Dados")
create_icon_button(left_frame, 'rotulo.png', lambda: apply_filter(label_connected_components), "Rotulagem")
# Inicializar a imagem como None
image = None
img_cv = None

root.mainloop()
