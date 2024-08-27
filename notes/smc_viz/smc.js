import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const stroke_width = 2;
const arrow_width = stroke_width * 2;
const url_params = new URLSearchParams(window.location.search);
const frame_background = d3.select("#frame-background");
const frame_foreground = d3.select("#frame-foreground");
const zoom_control = d3.zoom().on("zoom", (e) => frame_foreground.attr("transform", e.transform));
const yspace_slider = document.getElementById('yspace_slider');
const yspace_value = document.getElementById('yspace_value');
const charWidth_slider = document.getElementById('charWidth_slider');
const charWidth_value = document.getElementById('charWidth_value');
const untangle_checkbox = document.getElementById("untangle_checkbox");
const toggle_collapsed_button = document.getElementById("toggle_collapsed_button");

let collapsed = false;
let current_data = null;
let leftInfoChoiceCollapsed = null;
let leftInfoChoiceExpanded = null;
let particle_yspace = 15;
let particle_charWidth = 8;

if (url_params.has("path")) {
  document.getElementById("load-path").value = url_params.get("path");
}
loadByPath();

frame_background.call(zoom_control);
document.getElementById("fileInput").addEventListener("change", () => loadFile(fileInput.files[0]));
window.loadByPath = loadByPath;
yspace_slider.addEventListener('input', updateYSpace);
charWidth_slider.addEventListener('input', update_charWidth);
untangle_checkbox.addEventListener("change", setUntangle);
toggle_collapsed_button.addEventListener("click", setToggleCollapsed);
document.getElementById('left-info-select').addEventListener('change', updateLeftInfo);

function loadFile(selectedFile) {
  const reader = new FileReader();
  reader.onload = (event) => {
    // Update the load-path textbox with the selected file name
    document.getElementById("load-path").value = selectedFile.name;
    // Actually load the file and display it
    loadByPath();
  };
  reader.readAsText(selectedFile);
}

function updateToggleButtonText() {
  toggle_collapsed_button.textContent = collapsed ? "Expand" : "Collapse";
}

function updateLeftInfo() {
  const leftInfoSelect = document.getElementById('left-info-select');
  if (collapsed) {
    leftInfoChoiceCollapsed = leftInfoSelect.value;
  } else {
    leftInfoChoiceExpanded = leftInfoSelect.value;
  }
  if (current_data) {
    showData(current_data, { collapsed, left_info: collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded });
  }
}

function populateLeftInfoSelect({collapsed}) {
  const leftInfoSelect = document.getElementById('left-info-select');
  leftInfoSelect.innerHTML = ''; // Clear existing options

  if (current_data && current_data.history && current_data.history.length > 0) {
    // Add "None" option
    const noneOption = document.createElement('option');
    noneOption.value = "";
    noneOption.textContent = "None";
    leftInfoSelect.appendChild(noneOption);

    const options = collapsed 
      ? Object.keys(current_data.history[0]).filter(key => key !== 'particles')
      : Object.keys(current_data.history[0].particles[0]);

    options.forEach(option => {
      const optionElement = document.createElement('option');
      optionElement.value = option;
      optionElement.textContent = option.charAt(0).toUpperCase() + option.slice(1).replace('_', ' ');
      leftInfoSelect.appendChild(optionElement);
    });

    // Set the default or previously selected value
    leftInfoSelect.value = collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded;
  }
}

function loadByPath(path) {
  path = path || document.getElementById("load-path").value;
  d3.json(`./${path}`, { cache: "no-store" })
    .then((data) => {
      current_data = data;
      showData(data, { collapsed, left_info: collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded });
      updateToggleButtonText();
      populateLeftInfoSelect({collapsed});
    })
    .catch((error) => console.log(error));
}

function updateYSpace() {
  particle_yspace = parseInt(yspace_slider.value);
  yspace_value.textContent = particle_yspace;
  if (current_data) {
    showData(current_data, {collapsed});
  }
}

function update_charWidth() {
  particle_charWidth = parseFloat(charWidth_slider.value);
  charWidth_value.textContent = particle_charWidth;
  if (current_data) {
    showData(current_data, {collapsed});
  }
}

function setUntangle() {
  if (current_data) {
    showData(current_data, { collapsed, untangle: untangle_checkbox.checked });
  }
}

function setToggleCollapsed() {
  collapsed = !collapsed;
  updateToggleButtonText();
  if (current_data) {
    // Reset the translation while keeping the zoom level
    const currentTransform = d3.zoomTransform(frame_background.node());
    const newTransform = d3.zoomIdentity.translate(0, 0).scale(currentTransform.k);
    frame_background.call(zoom_control.transform, newTransform);
    
    showData(current_data, { collapsed, left_info: collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded });
    populateLeftInfoSelect({collapsed});
    
    const leftInfoSelect = document.getElementById('left-info-select');
    leftInfoSelect.value = collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded;
  }
}

function clearHighlights() {
  frame_foreground.selectAll(".highlighted").classed("highlighted", false);
}

function highlightAncestors(history, particle, t, i) {
  console.log("highlighting ancestors of", "particle", i, "at time", t);
  // Remove previous highlights
  clearHighlights();

  // Traverse and highlight ancestors
  let currentParticle = particle;
  let currentT = t;
  let currentI = i;
  
  while (currentT > 0) {

    const linkId = `#link-${currentT}-${currentI}`;
    frame_foreground.select(linkId).classed("highlighted", true);

    const ancestorId = `#particle-${currentT}-${currentI}`;
    frame_foreground.select(ancestorId).classed("highlighted", true);

    currentT--;
    currentI = currentParticle.parent;
    currentParticle = history[currentT].particles[currentI];
  }

  const rootId = `#particle-${currentT}-${currentI}`;
  frame_foreground.select(rootId).classed("highlighted", true);

  frame_foreground.selectAll("path.highlighted").raise();
  frame_foreground.selectAll(".particle.highlighted").raise();
  frame_foreground.selectAll("text.highlighted").raise();

}

function showData(data, options = {}) {
  const {
    collapsed = false,
    untangle = untangle_checkbox.checked,
    left_info = collapsed ? leftInfoChoiceCollapsed : leftInfoChoiceExpanded
  } = options;

  const svg_margin = 20;
  const SVG_WIDTH = (window.innerWidth > 0 ? window.innerWidth : screen.width) - svg_margin * 2;
  const SVG_HEIGHT = (window.innerHeight > 0 ? window.innerHeight : screen.height) - 140;

  const history = data.history;
  window.svg = history;

  // modify the history data in place however we need
  history.forEach((step, t) => {
    step.is_resampled = step.resample_indices ? true : false;

    step.re_indices = step.resample_indices ? step.resample_indices :
      // add in re_indices if they don't exist (each particle is extended)
      step.re_indices = Array.from({length: step.particles.length}, (_, i) => i);

    // Untangling logic (so resampled indices don't need to be sorted in the json input)
    if (untangle && t > 0 && history[t - 1].is_resampled) {
        const prevStep = history[t - 1];
        const perm = createPermutation(prevStep.re_indices);
        
        // Reorder previous step's re_indices and current step's particles
        prevStep.re_indices = perm.permute(prevStep.re_indices);
        step.particles = perm.permute(step.particles);
        
        // Update current step's re_indices
        step.re_indices = perm.reIndex(step.re_indices);
      }
  });

  history.forEach((step, t) => {
    const particles = step.particles;
    step.weight_total = particles.reduce((acc, p) => logaddexp(acc, p.weight), -Infinity);

    particles.forEach((particle, i) => {
      particle.relative_weight = Math.exp(particle.weight - step.weight_total);
      particle.prefix = particle.context.slice(0, -1).join("");
      particle.token = particle.context.slice(-1).join("");
      if (t > 0) {
        particle.parent = history[t - 1].is_resampled ? history[t - 1].re_indices[i] : i;
      }
      
      if (untangle && step.is_resampled) {
        const perm = createPermutation(step.re_indices);
        
        // Reorder step's re_indices
        step.re_indices = perm.permute(step.re_indices);
      }

      particle.children = step.re_indices
        .map((v, j) => v === i ? j : null) // find locations where i is resampled
        .filter(e => e !== null);
      
    });
  })

  const longest_token_length = Math.max(2, ...history.flatMap(step => step.particles.map(particle => particle.token.length)));
  const particle_xspace = collapsed ? particle_charWidth * longest_token_length : particle_yspace;
  const left_space = 100;
  const x_offset = (left_info ? left_space : 0) + particle_xspace;
  const y_offset = particle_yspace;

  const svg = d3.select("#svg svg");
  svg.classed("collapsed", collapsed); // Add or remove 'collapsed' class

  frame_foreground.selectAll("*").remove();
  frame_background.attr("width", SVG_WIDTH).attr("height", SVG_HEIGHT);
  frame_background.on("click", clearHighlights);

  const link = d3.linkVertical().x(d => d.x).y(d => d.y);
  const tooltip = d3.select("body").append("div").attr("class", "tooltip");

  const step_groups = frame_foreground.selectAll("g.step")
    .data(history)
    .enter()
    .append("g")
    .attr("class", "step")
    .attr("transform", (d, t) => `translate(0, ${t * (collapsed ? 2 * y_offset : particle_yspace * d.particles.length + y_offset)})`);

  // Create a group for paths so that we can manipulate separately
  const pathGroup = frame_foreground.append("g").attr("class", "path-group");

  step_groups.each(function (step, t) {
    const step_group = d3.select(this);
    const largest_relweight = step.particles.reduce((acc, p) => Math.max(acc, p.relative_weight), 0);
    const num_particles = step.particles.length;
    const step_yspace = collapsed ? 2 * y_offset : particle_yspace * num_particles + y_offset;
    const step_xspace = particle_xspace * num_particles;

    const color_scale = d3.scaleLinear().domain([0, num_particles - 1]).range(["red", "blue"]);

    const particle_groups = step_group.selectAll("g.particle")
      .data(step.particles)
      .enter()
      .append("g")
      .attr("class", "particle")
      .attr("transform", (d, i) => `translate(${x_offset + i * particle_xspace}, ${y_offset + (collapsed ? 0 : i * particle_yspace)})`)

    particle_groups.each(function (particle, i) {
      const particle_g = d3.select(this);
      const r = particle_yspace / 2;
      const max_length = 300;
      const radius = largest_relweight > 0 && !isNaN(particle.relative_weight)
        ? r * Math.sqrt(particle.relative_weight / largest_relweight) + 2
        : 1;

      particle_g.append("circle")
        .attr("r", radius)
        .attr("fill", color_scale(i))
        
      particle_g
        .attr("id", `particle-${t}-${i}`)
        .on("click", (event, d) => highlightAncestors(history, d, t, i));

      if (collapsed) {
        const particle_text = particle_g.append("text")
        particle_text.append("tspan")
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("class", "token")
          .text(particle.token);

        if (left_info) {
          let leftInfo = step[left_info] || "";
          if (typeof leftInfo === 'number' && !Number.isInteger(leftInfo)) {
            leftInfo = leftInfo.toExponential(4);
          }
          // display left info to the left of step_group
          step_group.append("text")
            .attr("x", 10)
            .attr("y", particle_yspace)
            // .attr("text-anchor", "end")
            .attr("dominant-baseline", "central")
            .append("tspan")
            .attr("class", "left-info")
            .text(leftInfo);
          step_group.append("line")
          .attr("x1", 1)
          .attr("x2", 100)
          .attr("y", particle_yspace)
          .attr("class", "particle-line")
          .lower();
        }
      } else {
        // Add text with prefix and token
        const particle_text = particle_g.append("text")
          .attr("x", step_xspace - i * particle_xspace)
          .attr("y", r / 2)
          .attr("text-anchor", "start");

        particle_text.append("tspan")
          .text(particle.prefix || "");

        particle_text.append("tspan")
          .attr("class", "token")
          .text(particle.token);

        // Add particle line
        particle_g.append("line")
          .attr("x1", 0)
          .attr("x2", step_xspace - i * particle_xspace)
          .attr("class", "particle-line")
          .lower();

        if (left_info) {
          let leftInfo = particle[left_info] || "";
          if (typeof leftInfo === 'number' && !Number.isInteger(leftInfo)) {
            leftInfo = leftInfo.toExponential(4);
          }
          particle_g.append("text")
            .attr("x", (num_particles - (i + 1)) * particle_xspace - step_xspace)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "central")
            .append("tspan")
            .attr("class", "left-info")
            .text(leftInfo);
          particle_g.append("line")
            .attr("x1", 0)
            .attr("x2", (num_particles - (i + 1)) * particle_xspace - step_xspace)
            .attr("class", "particle-line")
            .lower();
        }
      }

      particle_g.on("mouseover", (event) => {
        tooltip.transition().duration(100).style("opacity", 0.9);
        const displayed_keys = ['token', 'prefix', 'weight', 'relative_weight', 'parent', 'context'];
        
        function formatValue(key, value, spanClassName = 'array-element') {
          if (Array.isArray(value)) {
            return value.map(item => `<span class="${spanClassName}">${item}</span>`).join('');
          }
          if (key === 'weight') return value.toExponential(2);
          if (key === 'relative_weight') return value.toFixed(4);
          return value;
        }

        const formatKeyValue = (key, value) => {
          const formattedKey = key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ');
          return `<div class="key-value-pair">
            <span class="key">${formattedKey}</span>
            <span class="value">${formatValue(key, value)}</span>
          </div>`;
        };

        tooltip
          .html(
            `
          <div class="title">Step ${t}, particle ${i}</div>
          ${displayed_keys.map(key => formatKeyValue(key, particle[key])).join('')}
          ${Object.entries(particle)
            .filter(([key]) => !displayed_keys.includes(key))
            .map(([key, value]) => formatKeyValue(key, value))
            .join('')}
        `
          )
          .style("left", event.pageX + 10 + "px")
          .style("top", event.pageY - 28 + "px");
      })
      .on("mouseout", () => {
        tooltip.transition().duration(500).style("opacity", 0);
      });

      if (t > 0) {
        const parent_x = particle.parent * particle_xspace;
        const parent_y = collapsed ? 0 : particle.parent * particle_yspace;
        const current_y = collapsed ? 0 : i * particle_yspace;
        
        const parent_state_y = (t - 1) * (collapsed ? 2 * y_offset : particle_yspace * history[t-1].particles.length + y_offset);
        const current_state_y = t * (collapsed ? 2 * y_offset : particle_yspace * num_particles + y_offset);

        const parent_location = {
          x: x_offset + parent_x,
          y: y_offset + parent_y + parent_state_y,
        };
        const current_location = {
          x: x_offset + i * particle_xspace,
          y: y_offset + current_y + current_state_y,
        };
        
        pathGroup
          .append("path")
          .attr(
            "d", link({
              source: parent_location,
              target: {
                x: current_location.x,
                y: current_location.y - radius - stroke_width * arrow_width,
              }})
          )
          .attr("id", `link-${t}-${i}`)
          .attr("fill", "none")
          .attr("stroke-width", stroke_width)
          .attr("stroke", color_scale(particle.parent))
          .attr("opacity", 0.3)
          .attr("marker-end", "url(#arrow)");
      }

    });

    // Add one more step to show resampling
    if (t === history.length - 1) {
      const next_step_y = (t + 1) * (collapsed ? 2 * y_offset : particle_yspace * num_particles + y_offset);
      
      step.particles.forEach((particle, i) => {
        const current_location = {
          x: x_offset + i * particle_xspace,
          y: y_offset + (collapsed ? 0 : i * particle_yspace) + t * (collapsed ? 2 * y_offset : particle_yspace * num_particles + y_offset),
        };
        
        // Draw a link for each child of the current particle
        particle.children.forEach(childIndex => {
          const placeholder_location = {
            x: x_offset + childIndex * particle_xspace,
            y: y_offset + (collapsed ? 0 : childIndex * particle_yspace) + next_step_y,
          };
          
          pathGroup
            .append("path")
            .attr(
              "d", link({
                source: current_location,
                target: {
                  x: placeholder_location.x,
                  y: placeholder_location.y - stroke_width * arrow_width,
                }})
            )
            .attr("id", `link-placeholder-${i}-${childIndex}`)
            .attr("fill", "none")
            .attr("stroke-width", stroke_width)
            .attr("stroke", color_scale(i))
            .attr("opacity", 0.3)
            .attr("marker-end", "url(#arrow)");
        });
      });
    }
  });

  pathGroup.lower();
  frame_foreground.selectAll(".particle").raise();
  frame_foreground.selectAll("text").raise();
}

function showProb(prob, digits = 0) {
  if (prob == 0) return "0";
  if (prob == 1) return "1";
  if (prob >= 1e-3 && prob <= 1e3)
    return prob.toPrecision(Math.max(digits, 1));
  return prob.toExponential(digits);
}

/**
 * Creates a permutation object based on the input array.
 * 
 * @param {Array} arr - The input array to create the permutation from.
 * @returns An object containing:
 *   - fromIndex: Array of indices sorted by the values in the input array.
 *   - toIndex: Array mapping the sorted indices back to their original positions.
 *   - permute: Function to apply the permutation to an array.
 *   - reIndex: Function to apply the inverse permutation to update an array of indices.
 */
function createPermutation(arr) {
  const fromIndex = Array.from(arr.keys()).sort((a, b) => arr[a] - arr[b]);
  const toIndex = fromIndex.map((_, i) => fromIndex.indexOf(i));
  
  // Utility functions
  const permute = (array) => fromIndex.map(i => array[i]);
  const reIndex = (indices) => indices.map(i => toIndex[i]);
  
  return { fromIndex, toIndex, permute, reIndex };
}

function logaddexp(x, y) {
  if (x === -Infinity) return y;
  if (y === -Infinity) return x;
  return Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)));
}

