"""
RBY1 Monitor WebUI 템플릿
record_rby1_standalone.py에서 사용하는 HTML/CSS/JS 템플릿

참고: 실제 임계값은 record_rby1_standalone.py의 TeleopSettings에서 전달됨
      아래 DEFAULT_LIMITS는 독립 테스트용 fallback 값
"""

# 독립 테스트용 기본값 (실제로는 TeleopSettings에서 전달됨)
# 마스터 암 XM430/XM540 기준
DEFAULT_LIMITS = {
    "temp_warning": 60,
    "temp_critical": 70,
    "current_warning": 0.8,   # XM430 연속 ~0.5A 기준
    "current_critical": 1.5,  # XM430 Stall 2.3A의 65%
    "torque_warning": 0.8,    # XM430 연속 0.6Nm 기준
    "torque_critical": 1.5,   # XM430 Stall 3.0Nm의 50%
}

# CSS 스타일
STYLES = """
* { box-sizing: border-box; }
body { 
    font-family: 'Segoe UI', Arial, sans-serif; 
    background: #1a1a2e; 
    color: #eee;
    margin: 0;
    padding: 20px;
}
h1 { color: #4fc3f7; margin-bottom: 10px; }
h2 { color: #81c784; margin: 15px 0 10px 0; font-size: 1.1em; }
.header { 
    display: flex; 
    justify-content: space-between; 
    align-items: center;
    margin-bottom: 20px;
}
.status-badge { 
    background: #2d2d44; 
    padding: 8px 16px; 
    border-radius: 20px;
    font-size: 0.9em;
}
.status-badge.ok { border-left: 3px solid #4caf50; }
.status-badge.warning { border-left: 3px solid #ff9800; }
.status-badge.error { border-left: 3px solid #f44336; }

.grid { 
    display: grid; 
    grid-template-columns: 1fr 1fr; 
    gap: 20px; 
}
@media (max-width: 1200px) {
    .grid { grid-template-columns: 1fr; }
}

.panel { 
    background: #252542; 
    padding: 15px; 
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.panel-title { 
    font-size: 1em;
    font-weight: bold;
    color: #4fc3f7;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #3d3d5c;
}

.cameras { display: flex; flex-direction: row; gap: 15px; overflow-x: auto; }
.camera { background: #1e1e38; padding: 10px; border-radius: 8px; flex-shrink: 0; }
.camera h3 { margin: 0 0 8px 0; color: #81c784; font-size: 0.9em; }
.camera img { max-width: 320px; border-radius: 4px; }

.motor-table { 
    width: 100%; 
    border-collapse: collapse; 
    font-size: 0.85em;
}
.motor-table th, .motor-table td { 
    padding: 6px 8px; 
    text-align: center;
    border-bottom: 1px solid #3d3d5c;
}
.motor-table th { 
    background: #1e1e38; 
    color: #aaa;
    font-weight: normal;
}
.motor-table tr:hover { background: #2d2d50; }

.val { font-family: 'Consolas', monospace; }
.val-ok { color: #81c784; }
.val-warn { color: #ffb74d; }
.val-critical { color: #ef5350; font-weight: bold; }

.alarm-btn {
    padding: 8px 16px;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.9em;
    margin-right: 10px;
    transition: all 0.3s;
}
.alarm-btn.on {
    background: #4caf50;
    color: white;
}
.alarm-btn.off {
    background: #555;
    color: #aaa;
}
.alarm-btn:hover {
    opacity: 0.8;
}

.bar-container {
    width: 60px;
    height: 8px;
    background: #1e1e38;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-left: 5px;
}
.bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.bar-ok { background: linear-gradient(90deg, #4caf50, #81c784); }
.bar-warn { background: linear-gradient(90deg, #ff9800, #ffb74d); }
.bar-critical { background: linear-gradient(90deg, #f44336, #ef5350); }

.btn-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 5px;
}
.btn-on { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
.btn-off { background: #555; }

.ma-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}
.ma-arm {
    background: #1e1e38;
    padding: 12px;
    border-radius: 8px;
}
.ma-arm h4 {
    margin: 0 0 10px 0;
    color: #4fc3f7;
    font-size: 0.95em;
}

#update-time { color: #888; font-size: 0.8em; }
"""

# JavaScript 코드
def get_javascript(limits: dict) -> str:
    """JavaScript 코드 생성 (관절별 임계값 지원)"""
    import json
    
    # 관절별 임계값 배열 (14개: right 7 + left 7)
    # 기본값: XM430 기준
    default_per_joint = [0.8]*14
    
    curr_warn_arr = limits.get('current_warning_per_joint', default_per_joint)
    curr_crit_arr = limits.get('current_critical_per_joint', [1.5]*14)
    torq_warn_arr = limits.get('torque_warning_per_joint', [0.8]*14)
    torq_crit_arr = limits.get('torque_critical_per_joint', [1.5]*14)
    
    return f"""
// 온도 임계값 (모든 모터 동일)
const TEMP_WARN = {limits.get('temp_warning', 60)};
const TEMP_CRIT = {limits.get('temp_critical', 70)};

// 관절별 임계값 배열 (14개: r_arm 0-6, l_arm 0-6)
// XM540 (관절 0-2): 고토크, XM430 (관절 3-6): 저토크
const CURR_WARN_ARR = {json.dumps(list(curr_warn_arr))};
const CURR_CRIT_ARR = {json.dumps(list(curr_crit_arr))};
const TORQ_WARN_ARR = {json.dumps(list(torq_warn_arr))};
const TORQ_CRIT_ARR = {json.dumps(list(torq_crit_arr))};

// 하위 호환성: 단일 값 (표시용)
const CURR_WARN = {limits.get('current_warning', 1.5)};
const CURR_CRIT = {limits.get('current_critical', 2.5)};
const TORQ_WARN = {limits.get('torque_warning', 2.0)};
const TORQ_CRIT = {limits.get('torque_critical', 3.0)};

let audioCtx = null;
let alarmEnabled = true;
let lastAlarmTime = 0;
const ALARM_COOLDOWN = 3000;

function playWarningBeep(type) {{
    if (!alarmEnabled) return;
    const now = Date.now();
    if (now - lastAlarmTime < ALARM_COOLDOWN) return;
    lastAlarmTime = now;
    
    try {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        if (type === 'critical') {{
            oscillator.frequency.value = 880;
            gainNode.gain.value = 0.3;
            oscillator.start();
            setTimeout(() => oscillator.stop(), 150);
            setTimeout(() => {{
                const osc2 = audioCtx.createOscillator();
                const gain2 = audioCtx.createGain();
                osc2.connect(gain2);
                gain2.connect(audioCtx.destination);
                osc2.frequency.value = 880;
                gain2.gain.value = 0.3;
                osc2.start();
                setTimeout(() => osc2.stop(), 150);
            }}, 200);
            setTimeout(() => {{
                const osc3 = audioCtx.createOscillator();
                const gain3 = audioCtx.createGain();
                osc3.connect(gain3);
                gain3.connect(audioCtx.destination);
                osc3.frequency.value = 880;
                gain3.gain.value = 0.3;
                osc3.start();
                setTimeout(() => osc3.stop(), 150);
            }}, 400);
        }} else {{
            oscillator.frequency.value = 440;
            gainNode.gain.value = 0.2;
            oscillator.start();
            setTimeout(() => oscillator.stop(), 300);
        }}
    }} catch (e) {{
        console.warn('Audio playback failed:', e);
    }}
}}

function getTempClass(temp) {{
    if (temp >= TEMP_CRIT) return 'val-critical';
    if (temp >= TEMP_WARN) return 'val-warn';
    return 'val-ok';
}}

function getBarClass(temp) {{
    if (temp >= TEMP_CRIT) return 'bar-critical';
    if (temp >= TEMP_WARN) return 'bar-warn';
    return 'bar-ok';
}}

// 관절별 전류 임계값 사용 (j: 0-13 관절 인덱스)
function getCurrentClass(curr, j) {{
    const crit = CURR_CRIT_ARR[j] || CURR_CRIT;
    const warn = CURR_WARN_ARR[j] || CURR_WARN;
    if (curr >= crit) return 'val-critical';
    if (curr >= warn) return 'val-warn';
    return 'val-ok';
}}

function getCurrentBarClass(curr, j) {{
    const crit = CURR_CRIT_ARR[j] || CURR_CRIT;
    const warn = CURR_WARN_ARR[j] || CURR_WARN;
    if (curr >= crit) return 'bar-critical';
    if (curr >= warn) return 'bar-warn';
    return 'bar-ok';
}}

// 관절별 토크 임계값 사용 (j: 0-13 관절 인덱스)
function getTorqueClass(torq, j) {{
    const crit = TORQ_CRIT_ARR[j] || TORQ_CRIT;
    const warn = TORQ_WARN_ARR[j] || TORQ_WARN;
    if (torq >= crit) return 'val-critical';
    if (torq >= warn) return 'val-warn';
    return 'val-ok';
}}

function getTorqueBarClass(torq, j) {{
    const crit = TORQ_CRIT_ARR[j] || TORQ_CRIT;
    const warn = TORQ_WARN_ARR[j] || TORQ_WARN;
    if (torq >= crit) return 'bar-critical';
    if (torq >= warn) return 'bar-warn';
    return 'bar-ok';
}}

function updateStatus() {{
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {{
            let hasCritical = false;
            let hasWarning = false;
            
            // 양팔 인덱스 (6~19: r_arm 6-12, l_arm 13-19)
            const armIndices = [6,7,8,9,10,11,12,13,14,15,16,17,18,19];
            
            // 온도 체크 (양팔만)
            if (data.robot.temperature && data.robot.temperature.length > 0) {{
                for (const idx of armIndices) {{
                    const temp = data.robot.temperature[idx] || 0;
                    if (temp >= TEMP_CRIT) hasCritical = true;
                    else if (temp >= TEMP_WARN) hasWarning = true;
                }}
            }}
            
            // 전류 체크 (양팔만, 관절별 임계값)
            if (data.robot.current && data.robot.current.length > 0) {{
                for (let j = 0; j < armIndices.length; j++) {{
                    const idx = armIndices[j];
                    const curr = data.robot.current[idx] || 0;
                    const currCrit = CURR_CRIT_ARR[j] || CURR_CRIT;
                    const currWarn = CURR_WARN_ARR[j] || CURR_WARN;
                    if (curr >= currCrit) hasCritical = true;
                    else if (curr >= currWarn) hasWarning = true;
                }}
            }}
            
            // 토크 체크 (양팔만, 관절별 임계값)
            if (data.robot.torque && data.robot.torque.length > 0) {{
                for (let j = 0; j < armIndices.length; j++) {{
                    const idx = armIndices[j];
                    const torq = data.robot.torque[idx] || 0;
                    const torqCrit = TORQ_CRIT_ARR[j] || TORQ_CRIT;
                    const torqWarn = TORQ_WARN_ARR[j] || TORQ_WARN;
                    if (torq >= torqCrit) hasCritical = true;
                    else if (torq >= torqWarn) hasWarning = true;
                }}
            }}
            
            // 알람 재생
            if (hasCritical) {{
                playWarningBeep('critical');
                document.body.style.borderTop = '4px solid #f44336';
            }} else if (hasWarning) {{
                playWarningBeep('warning');
                document.body.style.borderTop = '4px solid #ff9800';
            }} else {{
                document.body.style.borderTop = '4px solid #4caf50';
            }}
            
            // 로봇 관절 상태 (양팔만 표시)
            const tbody = document.getElementById('robot-joints');
            if (data.robot.temperature && data.robot.temperature.length > 0) {{
                let html = '';
                const jointNames = ['r_arm_0','r_arm_1','r_arm_2','r_arm_3','r_arm_4','r_arm_5','r_arm_6',
                                  'l_arm_0','l_arm_1','l_arm_2','l_arm_3','l_arm_4','l_arm_5','l_arm_6'];
                for (let j = 0; j < armIndices.length; j++) {{
                    const i = armIndices[j];
                    const temp = data.robot.temperature[i] || 0;
                    const curr = data.robot.current[i] || 0;
                    const torq = data.robot.torque[i] || 0;
                    const name = jointNames[j] || 'joint_' + j;
                    
                    const tempClass = getTempClass(temp);
                    const tempBarClass = getBarClass(temp);
                    const tempBarWidth = Math.min(100, (temp / TEMP_CRIT) * 100);
                    
                    // 관절별 임계값으로 바 너비 계산
                    const currCrit = CURR_CRIT_ARR[j] || CURR_CRIT;
                    const torqCrit = TORQ_CRIT_ARR[j] || TORQ_CRIT;
                    
                    const currClass = getCurrentClass(curr, j);
                    const currBarClass = getCurrentBarClass(curr, j);
                    const currBarWidth = Math.min(100, (curr / currCrit) * 100);
                    
                    const torqClass = getTorqueClass(torq, j);
                    const torqBarClass = getTorqueBarClass(torq, j);
                    const torqBarWidth = Math.min(100, (torq / torqCrit) * 100);
                    
                    html += `<tr>
                        <td>${{name}}</td>
                        <td><span class="val ${{tempClass}}">${{temp.toFixed(0)}}</span>
                            <div class="bar-container"><div class="bar ${{tempBarClass}}" style="width:${{tempBarWidth}}%"></div></div>
                        </td>
                        <td><span class="val ${{currClass}}">${{curr.toFixed(2)}}</span>
                            <div class="bar-container"><div class="bar ${{currBarClass}}" style="width:${{currBarWidth}}%"></div></div>
                        </td>
                        <td><span class="val ${{torqClass}}">${{torq.toFixed(2)}}</span>
                            <div class="bar-container"><div class="bar ${{torqBarClass}}" style="width:${{torqBarWidth}}%"></div></div>
                        </td>
                    </tr>`;
                }}
                tbody.innerHTML = html;
            }}
            
            // 마스터 암 상태
            document.getElementById('btn-right').className = 
                'btn-indicator ' + (data.master_arm.button_right ? 'btn-on' : 'btn-off');
            document.getElementById('btn-left').className = 
                'btn-indicator ' + (data.master_arm.button_left ? 'btn-on' : 'btn-off');
            document.getElementById('trigger-right').textContent = data.master_arm.trigger_right;
            document.getElementById('trigger-left').textContent = data.master_arm.trigger_left;
            
            // 마스터 암 관절
            if (data.master_arm.q_joint && data.master_arm.q_joint.length >= 14) {{
                let rightHtml = '';
                let leftHtml = '';
                for (let i = 0; i < 7; i++) {{
                    rightHtml += `<tr><td>${{i}}</td><td class="val">${{data.master_arm.q_joint[i].toFixed(3)}}</td></tr>`;
                    leftHtml += `<tr><td>${{i}}</td><td class="val">${{data.master_arm.q_joint[i+7].toFixed(3)}}</td></tr>`;
                }}
                document.getElementById('ma-right-joints').innerHTML = rightHtml;
                document.getElementById('ma-left-joints').innerHTML = leftHtml;
            }}
            
            // 그리퍼 상태
            const gripperTbody = document.getElementById('gripper-data');
            if (data.gripper.connected && data.gripper.target_q && data.gripper.target_q.length > 0) {{
                let html = '';
                for (let i = 0; i < data.gripper.target_q.length; i++) {{
                    const target = data.gripper.target_q[i];
                    const min = data.gripper.min_q[i] || 0;
                    const max = data.gripper.max_q[i] || 1;
                    const range = max - min;
                    const progress = range > 0 ? ((target - min) / range * 100) : 0;
                    html += `<tr>
                        <td>${{i}}</td>
                        <td class="val">${{target.toFixed(1)}}</td>
                        <td class="val">${{min.toFixed(1)}}</td>
                        <td class="val">${{max.toFixed(1)}}</td>
                        <td><div class="bar-container" style="width:80px;">
                            <div class="bar bar-ok" style="width:${{progress.toFixed(0)}}%"></div>
                        </div></td>
                    </tr>`;
                }}
                gripperTbody.innerHTML = html;
            }} else {{
                gripperTbody.innerHTML = '<tr><td colspan="5" style="color:#888;">Not connected</td></tr>';
            }}
            
            // 업데이트 시간
            document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
        }})
        .catch(e => {{
            document.getElementById('connection-status').className = 'status-badge error';
            document.getElementById('connection-status').textContent = '● Disconnected';
        }});
}}

function toggleAlarm() {{
    alarmEnabled = !alarmEnabled;
    const btn = document.getElementById('alarm-toggle');
    if (alarmEnabled) {{
        btn.className = 'alarm-btn on';
        btn.textContent = '🔊 Alarm ON';
    }} else {{
        btn.className = 'alarm-btn off';
        btn.textContent = '🔇 Alarm OFF';
    }}
}}

// 초기 로드 및 주기적 업데이트
updateStatus();
setInterval(updateStatus, 500);
"""


def generate_html(camera_divs: str, stream_port: int, limits: dict = None) -> str:
    """전체 HTML 페이지 생성
    
    Args:
        camera_divs: 카메라 HTML (각 카메라별 <div>)
        stream_port: 스트리밍 포트 번호
        limits: 임계값 딕셔너리 (기본값 사용시 None)
    
    Returns:
        완전한 HTML 문자열
    """
    if limits is None:
        limits = DEFAULT_LIMITS
    
    if not camera_divs:
        camera_divs = '<p style="color: #ff9800;">카메라가 연결되지 않았습니다.</p>'
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>RBY1 Monitor</title>
    <style>
{STYLES}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 RBY1 Monitor</h1>
        <div>
            <button id="alarm-toggle" class="alarm-btn on" onclick="toggleAlarm()">🔊 Alarm ON</button>
            <span class="status-badge ok" id="connection-status">● Connected</span>
            <span id="update-time"></span>
        </div>
    </div>
    
    <!-- 카메라 패널 (전체 너비) -->
    <div class="panel" style="margin-bottom: 20px;">
        <div class="panel-title">📷 Cameras</div>
        <div class="cameras">
            {camera_divs}
        </div>
    </div>
    
    <div class="grid">
        <!-- 로봇 상태 + 그리퍼 패널 -->
        <div class="panel">
            <div class="panel-title">🦾 Robot Arms & Gripper</div>
            <div id="robot-status">
                <table class="motor-table">
                    <thead>
                        <tr>
                            <th>Joint</th>
                            <th>Temp (°C)</th>
                            <th>Current (A)</th>
                            <th>Torque (Nm)</th>
                        </tr>
                    </thead>
                    <tbody id="robot-joints">
                        <tr><td colspan="4" style="color:#888;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <!-- 그리퍼 (같은 패널 내) -->
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #3d3d5c;">
                <div style="color: #4fc3f7; font-weight: bold; margin-bottom: 10px;">✊ Gripper</div>
                <table class="motor-table">
                    <thead>
                        <tr><th>ID</th><th>Target</th><th>Min</th><th>Max</th><th>Progress</th></tr>
                    </thead>
                    <tbody id="gripper-data"></tbody>
                </table>
            </div>
        </div>
        
        <!-- 마스터 암 패널 -->
        <div class="panel">
            <div class="panel-title">🎮 Master Arm Status</div>
            <div class="ma-grid">
                <div class="ma-arm">
                    <h4><span class="btn-indicator btn-off" id="btn-right"></span>Right Arm</h4>
                    <div>Trigger: <span class="val" id="trigger-right">0</span></div>
                    <table class="motor-table" style="margin-top:8px;">
                        <thead><tr><th>J</th><th>Position (rad)</th></tr></thead>
                        <tbody id="ma-right-joints"></tbody>
                    </table>
                </div>
                <div class="ma-arm">
                    <h4><span class="btn-indicator btn-off" id="btn-left"></span>Left Arm</h4>
                    <div>Trigger: <span class="val" id="trigger-left">0</span></div>
                    <table class="motor-table" style="margin-top:8px;">
                        <thead><tr><th>J</th><th>Position (rad)</th></tr></thead>
                        <tbody id="ma-left-joints"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
{get_javascript(limits)}
    </script>
</body>
</html>'''


def generate_camera_div(cam_name: str, width: int = 480) -> str:
    """개별 카메라 div 생성"""
    return f'''
    <div class="camera">
        <h3>{cam_name}</h3>
        <img src="/{cam_name}.mjpeg" width="{width}">
    </div>'''
