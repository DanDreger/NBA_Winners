d3.json("/static/data/seasons.json").then((data) => {
  function setYears(){
    //Get Years
    let years = []
    for(const obj of data){
      years.push(Object.keys(obj))
    }
    //Add Years to options
    for(const year of years){
      let optionYear = d3.select("#selYear").append("option")
      optionYear.text(`${year}`)
      optionYear.attr("value", `${year}`)
    }
  }

  function setTeams(){

    //Get Year
    let selectedYear = d3.select("#selYear").property("value")

    //Get teams from respective year
    let teams = Object.keys(data[parseInt(selectedYear) - 2011][selectedYear])

    //Remove Teams that were previously set
    d3.select("#selTeam").html("");

    //Add Teams to options
    for(const team of teams){
      let optionTeam = d3.select("#selTeam").append("option")
      optionTeam.text(`${team}`)
      optionTeam.attr("value", `${team}`)
    }
    
  }
  function setRecords (mappedData){
    let wins = mappedData['win_loss']
    let playoffWin = mappedData["playoff_win_loss"]
    let superBowlWinner = mappedData["superbowl_winner"]
    d3.select("#teamrecord").text(`${wins["win"]} - ${wins["loss"]} - ${wins["tie"]}`)
    d3.select('#playoffteamrecord').text(Object.keys(playoffWin).length !== 0 && playoffWin.constructor === Object ?
     `${playoffWin["win"]} - ${playoffWin["loss"]} - ${playoffWin["tie"]}${superBowlWinner ? `, Super Bowl Winner`: ""}` : `Team did not make the playoffs.`)
  }
  function getId(positions) {
    let group = []
    for(const position of positions){
      if(["QB", "WR", "RB", "TE", "OL", "C", "FB", "G", "T"].includes(position)){
        group.push("Offensive Team")
      }else if (["DL", "DT", "DE", "OLB", "LB", "MLB", "CB", "S", "LT", "RT", "ILB", "FS", "LS", "RS", "SS"].includes(position)){
        group.push("Defensive Team")
      }else{
        group.push("Special Teams")
      }
    }

    return group;
  }

  function init(){
    let selYear="", selTeam = "";
    selYear = d3.select("#selYear").property("value")
    selTeam = d3.select("#selTeam").property("value")
    let parseYear = parseInt(selYear) - 2011
    let mappedData= data[parseYear][selYear][selTeam]
    let mapSalary = mappedData['players'].map((item) => item['cap_hit'])
    let mapName = mappedData['players'].map((item) => item['name'])
    let positions = mappedData['players'].map((item) => item['position'])
    let group = getId(positions)
    setRecords(mappedData)

    let toShow = [{
      values: mapSalary,
      labels: mapName,
      type: 'pie',
      hole: .3,
      title: selTeam,
      hovertemplate: 'Player Name: %{label} <br>Cap Hit: \$%{value} <br>Percentage: %{percent} <br>Position: %{customdata[0]} <br>Team: %{id}<extra></extra>',
      textposition: "inside",
      texttemplate: "%{percent}",
      showlegend: true,
      customdata: positions,
      ids: group,
      marker: {

        line: {
          color: "black",
          width: 0.3
        }
    
      },
    }];
    let layout = {
      autosize: true,
      legend : {
        "title": {
          "text": "<b>Player Names: (Highest to Lowest Cap Hit)</b>"
        },
        "itemclick": false,
        "itemdoubleclick":false,
        
      },
    };
    
    Plotly.newPlot("chart", toShow, layout, {responsive: true});
  }
  setYears()
  setTeams()
  init()
  d3.selectAll("#selYear").on("change", () => {
    setTeams();
    init();
  });
  d3.selectAll("#selTeam").on("change", () => {
    init();
  });
});