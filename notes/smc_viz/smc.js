import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const stroke_width = 2;
const arrow_width = stroke_width * 2;
const char_width = 8;

const url_params = new URLSearchParams(window.location.search);
if (url_params.has("path")) {
  document.getElementById("load-path").value = url_params.get("path");
}
loadByPath();

document.getElementById("fileInput").addEventListener("change", () => loadFile(fileInput.files[0]));

function loadFile(selectedFile) {
  const reader = new FileReader();
  reader.onload = (event) => {
    resetUntangleButton(); // Reset untangle button state
    // Update the load-path textbox with the selected file name
    document.getElementById("load-path").value = selectedFile.name;
    // Actually load the file and display it
    loadByPath();
  };
  reader.readAsText(selectedFile);
}

function logaddexp(x, y) {
  if (x === -Infinity) return y;
  if (y === -Infinity) return x;
  return Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)));
}

let collapsed = false;
let current_data = null;
const toggle_button = document.getElementById("toggle_view");

function updateToggleButtonText() {
  toggle_button.textContent = collapsed ? "Expand" : "Collapse";
}

let leftInfoChoice = 'weight'; // Default value

// Add this function to update the global left_info setting
function updateLeftInfo() {
  const leftInfoSelect = document.getElementById('left-info-select');
  leftInfoChoice = leftInfoSelect.value;
  if (current_data) {
    showData(current_data, { collapsed, left_info: leftInfoChoice });
  }
}

function loadByPath(path) {
  path = path || document.getElementById("load-path").value;
  d3.json(`./${path}`, { cache: "no-store" })
    .then((data) => {
      current_data = data;
      showData(data, { collapsed, left_info: leftInfoChoice });
      updateToggleButtonText();
      resetUntangleButton(); // Reset untangle button state
      populateLeftInfoSelect(); // Add this line to make left info select
    })
    .catch((error) => console.log(error));
}

window.loadByPath = loadByPath; // Add this line to attach the function to the global scope

function resetUntangleButton() {
  untangle_button.disabled = false;
  untangle_button.textContent = "Untangle";
}

const frame_background = d3.select("#frame-background");
const frame_foreground = d3.select("#frame-foreground");

const zoom_control = d3.zoom().on("zoom", (e) => frame_foreground.attr("transform", e.transform));
frame_background.call(zoom_control);

let particle_yspace = 15;
const yspace_slider = document.getElementById('yspace_slider');
const yspace_value = document.getElementById('yspace_value');

function updateYSpace() {
  particle_yspace = parseInt(yspace_slider.value);
  yspace_value.textContent = particle_yspace;
  if (current_data) {
    showData(current_data, {collapsed});
  }
}

yspace_slider.addEventListener('input', updateYSpace);

const untangle_button = document.getElementById("untangle_button");

function setUntangle() {
  untangle_button.disabled = true;
  untangle_button.textContent = "Untangled";
  if (current_data) {
    showData(current_data, { collapsed, untangle: true });
  }
}

untangle_button.addEventListener("click", setUntangle);

function showData(data, options = {}) {
  const {
    collapsed = false,
    untangle = false,
    left_info = leftInfoChoice
  } = options;

  const svg_margin = 20;
  const SVG_WIDTH = (window.innerWidth > 0 ? window.innerWidth : screen.width) - svg_margin * 2;
  const SVG_HEIGHT = (window.innerHeight > 0 ? window.innerHeight : screen.height) - 140;

  const history = data.history;
  window.svg = history;

  // modify the history data in place however we need
  history.forEach((step, t) => {

    // Untangling logic
    if (untangle && t > 0 && history[t - 1].resample_indices) {
        const prevStep = history[t - 1];
        const perm = createPermutation(prevStep.resample_indices);
        
        // Reorder previous step's resample_indices and current step's particles
        prevStep.resample_indices = perm.permute(prevStep.resample_indices);
        step.particles = perm.permute(step.particles);
        
        // Update current step's resample_indices if it exists
        if (step.resample_indices) {
            step.resample_indices = perm.reIndex(step.resample_indices);
        }
        
        // //Propagate reordering to all future steps (unnecessary, since only depends on prev step.)
        // for (let futureT = t + 1; futureT < history.length; futureT++) {
        //     const futureStep = history[futureT];
        //     futureStep.particles = perm.permute(futureStep.particles);
        //     if (futureStep.resample_indices) {
        //         futureStep.resample_indices = perm.reIndex(futureStep.resample_indices);
        //     }
        // }
    }


    const particles = step.particles;
    step.weight_total = particles.reduce((acc, p) => logaddexp(acc, p.weight), -Infinity);
    particles.forEach((particle, i) => {
      particle.relative_weight = Math.exp(particle.weight - step.weight_total);
      particle.prefix = particle.context.slice(0, -1).join("");
      particle.token = particle.context.slice(-1).join("");
      if (t > 0) {
        particle.parent = history[t - 1].resample_indices ? history[t - 1].resample_indices[i] : i;
      }
    });

  });

  const longest_token_length = Math.max(2, ...history.flatMap(step => step.particles.map(particle => particle.token.length)));
  const particle_xspace = collapsed ? char_width * longest_token_length : particle_yspace;
  const left_space = 100;
  const x_offset = (left_info && !collapsed ? left_space : 0) + particle_xspace;
  const y_offset = particle_yspace;

  const svg = d3.select("#svg svg");
  svg.classed("collapsed", collapsed); // Add or remove 'collapsed' class

  frame_foreground.selectAll("*").remove();
  frame_background.attr("width", SVG_WIDTH).attr("height", SVG_HEIGHT);

  const link = d3.linkVertical().x(d => d.x).y(d => d.y);
  const tooltip = d3.select("body").append("div").attr("class", "tooltip");

  const step_groups = frame_foreground.selectAll("g.step")
    .data(history)
    .enter()
    .append("g")
    .attr("class", "step")
    .attr("transform", (d, t) => `translate(0, ${t * (collapsed ? 2 * y_offset : particle_yspace * d.particles.length + y_offset)})`);

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
      .attr("transform", (d, i) => `translate(${x_offset + i * particle_xspace}, ${y_offset + (collapsed ? 0 : i * particle_yspace)})`);

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
        .classed("particle-circle", true);

      if (collapsed) {
        const particle_text = particle_g.append("text")
        particle_text.append("tspan")
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("class", "token")
          .text(particle.token);
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
        tooltip.transition().duration(200).style("opacity", 0.9);
        const displayed_keys = ['token', 'prefix', 'weight', 'relative_weight', 'parent'];
        const additionalInfo = Object.entries(particle)
        .filter(([key]) => !displayed_keys.includes(key))
          .map(([key, value]) => `<strong>${key}</strong>: ${value}<br/>`)
          .join('')
        tooltip
          .html(
            `
          <u>Step ${t}, particle ${i}:</u><br/>
          ${displayed_keys.map(key => {
              const val = particle[key];
              const formattedKey = key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ');
              return `<strong>${formattedKey}</strong>: ${
                key === 'weight' ? val.toExponential(2) :
                key === 'relative_weight' ? val.toFixed(4) :
                val
              }<br/>`;
            }).join('')}
          ${additionalInfo}
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
        
        frame_foreground
          .append("path")
          .attr(
            "d",
            link({
              source: parent_location,
              target: {
                x: current_location.x,
                y: current_location.y - radius - stroke_width * arrow_width,
              },
            })
          )
          .attr("fill", "none")
          .attr("stroke-width", stroke_width)
          .attr("stroke", color_scale(particle.parent))
          .attr("opacity", 0.3)
          .attr("marker-end", "url(#arrow)");
      }
    });
  });
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

toggle_button.addEventListener("click", () => {
  collapsed = !collapsed;
  updateToggleButtonText();
  if (current_data) {
    showData(current_data, { collapsed });;
  }
});

// Add this at the end of the file
// Populate the left-info select element with available options
function populateLeftInfoSelect() {
  const leftInfoSelect = document.getElementById('left-info-select');
  if (current_data && current_data.history && current_data.history[0] && current_data.history[0].particles[0]) {
    const particleKeys = Object.keys(current_data.history[0].particles[0]);
    leftInfoSelect.innerHTML = '';
    particleKeys.forEach(key => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = key;
      leftInfoSelect.appendChild(option);
    });
    leftInfoSelect.value = leftInfoChoice;
  }
}

// Call this function when data is loaded
window.addEventListener('load', () => {
  const leftInfoSelect = document.getElementById('left-info-select');
  leftInfoSelect.addEventListener('change', updateLeftInfo);
});
