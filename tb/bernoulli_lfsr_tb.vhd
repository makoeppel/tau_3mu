library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bernoulli_lfsr_tb is
end entity;

architecture arch of bernoulli_lfsr_tb is

    constant CLK_MHZ : real := 1000.0; -- MHz
    signal clk, reset_n : std_logic := '0';

    signal activation : std_logic_vector(9 downto 0);
    signal state_counter : integer := 0;
    type memory is array(0 to 5) of std_logic_vector(9 downto 0);
    constant rom : memory := (
        x"030", x"010", x"000", x"2f8", x"170", x"0b0", x"050",
        x"020", x"308", x"278", x"130", x"090", x"040", x"318",
        x"180", x"3b8", x"1d0", x"0e0", x"368", x"2a8", x"248",
        x"218", x"100", x"378", x"1b0", x"0d0", x"060", x"328",
        x"288", x"238", x"110", x"080", x"338", x"190", x"0c0",
        x"358", x"1a0", x"3c8", x"2d8", x"160", x"3a8", x"2c8",
        x"258", x"120", x"388", x"2b8", x"150", x"0a0", x"348",
        x"298", x"140", x"398", x"1c0", x"3d8", x"1e0", x"3e8",
        x"2e8", x"268", x"228", x"208", x"1f8", x"0f0"
    );

begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    e_bernoulli_lfsr : entity work.bernoulli_lfsr
    port map (
        i_sync_reset => not reset_n,
        i_seed => "0010000",
        i_en => '1',
        i_activation => activation,
        o_bernoulli => open,
        o_lfsr => open,

        i_reset_n => reset_n,
        i_clk => clk--,
    );

    process(clk, reset_n)
    begin
    if ( reset_n /= '1' ) then
        activation <= (others => '0');
        state_counter <= 20;
    elsif rising_edge(clk) then
        if ( state_counter = 61 ) then
            state_counter <= 0;
        else
            state_counter <= state_counter + 1;
        end if;
        activation <= rom(state_counter);
    end if;
    end process;

end architecture;
