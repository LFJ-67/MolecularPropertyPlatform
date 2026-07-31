let composer;

document
.getElementById("draw-btn")
.addEventListener("click",openEditor);



function openEditor(){

    document
    .getElementById("editor-dialog")
    .style.display="block";


    if(!composer){

        composer =
        new Kekule.Editor.Composer(
            document.getElementById("editor")
        );


        composer.setChemObj(
            Kekule.IO.loadFormatData(
                "",
                "smi"
            )
        );

    }

}

document
.getElementById("use-structure")
.addEventListener(
"click",
function(){

    let mol =
    composer.getChemObj();


    let smiles =
    Kekule.IO.saveFormatData(
        mol,
        "smi"
    );


    document
    .getElementById("smiles")
    .value=smiles;


    document
    .getElementById("editor-dialog")
    .style.display="none";

});


let currentTarget = "triplet";
const buttons = document.querySelectorAll(".nav-btn");

// 导航栏切换
buttons.forEach(button => {
    button.addEventListener("click", () => {
        buttons.forEach(b => b.classList.remove("active"));
        button.classList.add("active");
        currentTarget = button.dataset.target;
        document.getElementById("current-title").innerText = button.dataset.title;
        document.getElementById("result").innerHTML = "";
    });
});

// 按钮与回车事件绑定
document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("smiles").addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
        predict();
    }
});

/**
 * 异步请求后端生成二维结构图片及分子描述符
 */
async function drawStructure(smiles) {
    try {
        const response = await fetch("/structure", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ smiles: smiles })
        });
        const data = await response.json();

        if (data.success) {
            // 1. 渲染图片
            document.getElementById("mol-image").src = "data:image/png;base64," + data.image;

            // 2. 渲染 RDKit 描述符数据（修复拼写并安全赋值）
            const info = data.info;
            if (info) {
                document.getElementById("formula").innerText = info.formula || "-";
                document.getElementById("molwt").innerText = info.molwt || "-";
                document.getElementById("tpsa").innerText = info.tpsa || "-";
                document.getElementById("logp").innerText = info.logp || "-";
                document.getElementById("hba").innerText = info.hba || "-";
                document.getElementById("hbd").innerText = info.hbd || "-";
                document.getElementById("rotatable").innerText = info.rotatable || "-";
                document.getElementById("rings").innerText = info.rings || "-";
            }
        } else {
            console.error("生成结构失败:", data.message);
        }
    } catch (error) {
        console.error("请求结构接口时发生错误:", error);
    }
}

/**
 * 异步请求后端预测化学性质
 */
async function predict() {
    let smiles = document.getElementById("smiles").value.trim();

    if (smiles === "") {
        alert("请输入SMILES");
        return;
    }

    // 显示加载状态
    document.getElementById("loading").style.display = "block";
    document.getElementById("result").innerHTML = "";

    // 同时触发结构与描述符获取
    await drawStructure(smiles);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                smiles: smiles,
                target: currentTarget
            })
        });

        const data = await response.json();

        // 确保无论成功失败，都关闭加载动画
        document.getElementById("loading").style.display = "none";

        if (data.success) {
            const result = data.data;
            document.getElementById("result").innerHTML = `
                <div class="result-card">
                    <h3>${result.property}</h3>
                    <div class="value">
                        ${result.prediction} ${result.unit}
                    </div>
                    <div class="time">
                        推理时间：${result.time} s
                    </div>
                </div>
            `;
        } else {
            document.getElementById("result").innerHTML = `<p style="color:red;">${data.message}</p>`;
        }
    } catch (error) {
        document.getElementById("loading").style.display = "none";
        document.getElementById("result").innerHTML = `<p style="color:red;">网络请求失败，请检查后端后台服务</p>`;
        console.error("预测请求错误:", error);
    }
}


function showSingleMode(){

    document
    .getElementById("single-page")
    .style.display="block";

    document
    .getElementById("batch-page")
    .style.display="none";

    document
    .getElementById("single-mode-btn")
    .classList.add("active");

    document
    .getElementById("batch-mode-btn")
    .classList.remove("active");

}

function showBatchMode(){

    document
    .getElementById("single-page")
    .style.display="none";

    document
    .getElementById("batch-page")
    .style.display="block";

    document
    .getElementById("batch-mode-btn")
    .classList.add("active");

    document
    .getElementById("single-mode-btn")
    .classList.remove("active");

}




// 绑定批量上传按钮的点击事件
document.getElementById("upload-btn").addEventListener("click", uploadBatchFile);




// 全局变量：用于暂存后端返回的安全文件名
let uploadedFilename = "";

// 1. 绑定“上传预览”按钮事件
document.getElementById("upload-btn").addEventListener("click", uploadBatchFile);

// 2. 绑定“开始批量预测”按钮事件
document.getElementById("start-batch-predict-btn").addEventListener("click", startBatchPrediction);

/**
 * 第一步：上传文件并获取预览
 */
async function uploadBatchFile() {
    const fileInput = document.getElementById("batch-file");
    const file = fileInput.files[0];
    const previewTable = document.getElementById("preview-table");
    const interactiveZone = document.getElementById("batch-interactive-zone");
    const batchStatus = document.getElementById("batch-status");

    if (!file) {
        alert("请先选择一个 CSV 或 Excel 文件！");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // 重置界面状态
    previewTable.innerHTML = "<p style='color: #666;'>正在上传并解析文件，请稍候...</p>";
    interactiveZone.style.display = "none";
    batchStatus.innerHTML = "";
    uploadedFilename = "";

    try {
        // 请求后端 /upload 接口
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            // 保存后端返回的 secure_filename
            uploadedFilename = data.filename;

            // 显示性质选择与预测按钮区域
            interactiveZone.style.display = "block";

            // 渲染后端返回的 preview 前五行数据
            renderPreviewTable(data.columns, data.preview);
        } else {
            previewTable.innerHTML = `<p style='color:red;'>上传失败: ${data.message}</p>`;
        }
    } catch (error) {
        previewTable.innerHTML = "<p style='color:red;'>网络请求失败，请检查后端后台服务是否正常</p>";
        console.error("上传请求错误:", error);
    }
}

/**
 * 辅助函数：将 JSON 预览数据渲染为 HTML 表格
 */
function renderPreviewTable(columns, preview) {
    const previewTable = document.getElementById("preview-table");
    if (!preview || preview.length === 0) {
        previewTable.innerHTML = "<p>文件上传成功，但未读取到有效行数据。</p>";
        return;
    }

    let html = "<h3 style='margin-top:20px;'>数据预览 (前5行)</h3>";
    html += "<table style='border-collapse: collapse; width: 100%; text-align: left; margin-top: 10px;'><thead><tr style='background-color: #f2f2f2;'>";

    // 生成表头
    columns.forEach(col => {
        html += `<th style='border: 1px solid #ddd; padding: 8px;'>${col}</th>`;
    });
    html += "</tr></thead><tbody>";

    // 生成表体数据
    preview.forEach(row => {
        html += "<tr>";
        columns.forEach(col => {
            let cellValue = row[col] !== undefined && row[col] !== null ? row[col] : "";
            html += `<td style='border: 1px solid #ddd; padding: 8px;'>${cellValue}</td>`;
        });
        html += "</tr>";
    });

    html += "</tbody></table>";
    previewTable.innerHTML = html;
}

/**
 * 第二步：发送文件名和目标性质，执行批量预测
 */
async function startBatchPrediction() {
    const batchStatus = document.getElementById("batch-status");
    const targetSelect = document.getElementById("batch-target-select");

    if (!uploadedFilename) {
        alert("请先上传文件！");
        return;
    }

    const target = targetSelect.value;
    batchStatus.innerHTML = "<span style='color: #007bff;'>后端深度学习模型正在推理中，请耐心等待...</span>";

    try {
        // 请求后端 /batch_predict 接口
        const response = await fetch("/batch_predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: uploadedFilename,
                target: target
            })
        });
        const data = await response.json();

        if (data.success) {
            // 预测成功，拼接后端对应的 /download/<filename> 路由
            batchStatus.innerHTML = `
                <span style='color: green;'>√ 批量预测完成！</span>
                <a href="/download/${data.download}" style='margin-left: 20px; color: #007bff; font-weight: bold; text-decoration: underline;' download>
                    点击下载预测结果文件
                </a>
            `;
        } else {
            batchStatus.innerHTML = `<span style='color: red;'>预测失败: ${data.message}</span>`;
        }
    } catch (error) {
        batchStatus.innerHTML = "<span style='color: red;'>网络请求失败，无法完成批量预测</span>";
        console.error("批量预测请求错误:", error);
    }
}
