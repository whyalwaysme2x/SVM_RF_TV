const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "";

const FIELD_CONFIG = [
  { name: "spec_Brand", label: "Hãng sản xuất", type: "select" },
  { name: "spec_Display_Type", label: "Loại màn hình", type: "select" },
  { name: "spec_Backlight_Type", label: "Backlight", type: "select" },
  { name: "spec_Resolution", label: "Độ phân giải", type: "select" },
  { name: "spec_LED_Panel_Type", label: "Panel LED", type: "select" },
  { name: "spec_High_Dynamic_Range_HDR", label: "HDR", type: "select" },
  { name: "spec_ENERGY_STAR_Certified", label: "Energy Star", type: "select" },
  { name: "spec_Remote_Control_Type", label: "Remote", type: "select" },
  {
    name: "spec_Model_Year",
    label: "Model Year",
    type: "select", // Đổi từ number sang select để chỉ hiển thị giá trị hợp lệ
  },
  {
    name: "spec_Refresh_Rate",
    label: "Refresh Rate (Hz)",
    type: "select", // Đổi từ number sang select để chỉ hiển thị giá trị hợp lệ
  },
  {
    name: "spec_Screen_Size_Class",
    label: "Kích thước (inch)",
    type: "select", // Đổi từ number sang select để chỉ hiển thị giá trị hợp lệ
  },
];

const SAMPLE_PRODUCTS = [
  {
    name: "Sony 4K 65 inch",
    brand: "Sony",
    price: "$1,799",
    tag: "Premium",
  },
  {
    name: "Hisense 32 inch",
    brand: "Hisense",
    price: "$329",
    tag: "Entry",
  },
  {
    name: "Xiaomi A32",
    brand: "Xiaomi",
    price: "$299",
    tag: "Entry",
  },
  {
    name: "TCL 32T31",
    brand: "TCL",
    price: "$399",
    tag: "Mid",
  },
  {
    name: "Smart TV Design",
    brand: "Premium",
    price: "$1,299",
    tag: "Premium",
  },
];

const productGrid = document.getElementById("productGrid");
const formContainer = document.getElementById("formInputs");
const form = document.getElementById("advisorForm");
const resultCard = document.getElementById("advisorResult");

let featureOptions = {};

renderProducts();
buildFormFields();
bootstrap();

function renderProducts() {
  // Dùng cùng 1 ảnh cho tất cả sản phẩm
  const imagePath = 'ảnh/android-tivi-led-hisense-32-inch-32a4n-1-638685824829514383-700x467.jpg';
  const imageUrl = `/static/${encodeURIComponent(imagePath)}`;
  
  productGrid.innerHTML = SAMPLE_PRODUCTS.map(
    (item) => `
      <article class="product-card">
        <div class="product-image">
          <img src="${imageUrl}" alt="${item.name}" loading="lazy" />
        </div>
        <span class="tag">${item.tag}</span>
        <div class="product-card-content">
          <strong>${item.name}</strong>
          <p>${item.brand}</p>
          <footer>
            <span>${item.price}</span>
            <button class="ghost">Chi tiết</button>
          </footer>
        </div>
      </article>
    `
  ).join("");
}

function buildFormFields() {
  FIELD_CONFIG.forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "form-field";
    const label = document.createElement("label");
    label.textContent = field.label;
    label.htmlFor = field.name;

    // Tất cả các trường giờ đều dùng select để đảm bảo chỉ chọn giá trị hợp lệ từ dataset
    const input = document.createElement("select");
    input.innerHTML = `<option value="">Chọn...</option>`;
    input.id = field.name;
    input.name = field.name;
    input.required = true;

    wrapper.append(label, input);
    formContainer.appendChild(wrapper);
  });
}

async function bootstrap() {
  try {
    const res = await fetch(`${API_BASE}/api/options`);
    const data = await res.json();
    featureOptions = data.features || {};
    populateSelects();
  } catch (error) {
    console.error(error);
    resultCard.innerHTML = `
      <p class="eyebrow">Lỗi kết nối</p>
      <h3>Không thể tải meta</h3>
      <p>Chạy backend FastAPI: <code>uvicorn backend.api:app --reload</code></p>
    `;
  }
}

function populateSelects() {
  FIELD_CONFIG.filter((f) => f.type === "select").forEach((field) => {
    const select = document.getElementById(field.name);
    const options = featureOptions[field.name] || [];
    
    // Sort options for numeric fields
    let sortedOptions = options;
    if (field.name === "spec_Model_Year" || 
        field.name === "spec_Refresh_Rate" || 
        field.name === "spec_Screen_Size_Class") {
      sortedOptions = options.map(opt => typeof opt === 'number' ? opt : parseFloat(opt))
        .filter(opt => !isNaN(opt))
        .sort((a, b) => a - b);
    } else {
      sortedOptions = options.sort();
    }
    
    // Format display text for better UX
    select.innerHTML =
      `<option value="">Chọn...</option>` +
      sortedOptions.map((opt) => {
        let displayText = opt;
        // Format display text
        if (field.name === "spec_Refresh_Rate") {
          displayText = `${opt} Hz`;
        } else if (field.name === "spec_Screen_Size_Class") {
          displayText = `${opt}"`;
        } else if (field.name === "spec_Model_Year") {
          displayText = `${opt}`;
        }
        return `<option value="${opt}">${displayText}</option>`;
      }).join("");
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    // Convert numeric fields to proper types
    if (key === "spec_Model_Year" || key === "spec_Refresh_Rate") {
      payload[key] = parseInt(value, 10);
    } else if (key === "spec_Screen_Size_Class") {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  });
  setResultLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Prediction failed");
    const data = await res.json();
    paintResult(data.prediction);
  } catch (error) {
    resultCard.innerHTML = `
      <p class="eyebrow">Prediction error</p>
      <h3>Backend chưa chạy?</h3>
      <p>${error.message}</p>
    `;
  } finally {
    setResultLoading(false);
  }
});

function setResultLoading(state) {
  if (state) {
    resultCard.classList.add("loading");
    resultCard.innerHTML = `
      <p class="eyebrow">Đang dự đoán</p>
      <h3>Đợi xíu...</h3>
      <p>Pipeline SVM & RandomForest xử lý input.</p>
    `;
  } else {
    resultCard.classList.remove("loading");
  }
}

function paintResult(prediction) {
  const confidence = prediction.segment_confidence
    ? `${Math.round(prediction.segment_confidence * 100)}%`
    : "—";
  const [low, high] = prediction.price_range;

  resultCard.innerHTML = `
    <p class="eyebrow">TV Advisor</p>
    <h3>Phân khúc ${prediction.segment}</h3>
    <p class="price">$${prediction.suggested_price}</p>
    <p>Tự tin: ${confidence}</p>
    <p>Khoảng giá gợi ý: $${low} – $${high}</p>
  `;
}

