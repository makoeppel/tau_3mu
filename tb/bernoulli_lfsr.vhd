library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bernoulli_lfsr is
generic (
    g_m : positive := 7;
    g_poly : std_logic_vector := "1100000" -- x^7+x^6+1
);
port (
    i_sync_reset    : in    std_logic;
    i_seed          : in    std_logic_vector (g_m-1 downto 0) := "0010000";
    i_en            : in    std_logic;
    i_activation    : in    std_logic_vector (9 downto 0);
    o_bernoulli     : out   std_logic;
    o_lfsr          : out   std_logic_vector (g_m-1 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture rtl of bernoulli_lfsr is

    signal r_lfsr : std_logic_vector (g_m downto 1);
    signal w_mask : std_logic_vector (g_m downto 1);
    signal w_poly : std_logic_vector (g_m downto 1);

    -- NOTE: for now we just use pre-computed normed values
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
    signal dout : std_logic_vector(9 downto 0);

begin

    o_lfsr <= r_lfsr(g_m downto 1);
    w_poly <= g_poly;
    g_mask : for k in g_m downto 1 generate
        w_mask(k) <= w_poly(k) and r_lfsr(1);
    end generate;

    -- lfsr
    -- TODO: one can fill the ROM also in hardware
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        r_lfsr <= (others => '1');
    elsif rising_edge(i_clk) then
        if ( i_sync_reset = '1' ) then
            r_lfsr <= i_seed;
        elsif (i_en = '1') then
            r_lfsr <= '0' & r_lfsr(g_m downto 2) xor w_mask;
        end if;
    end if;
    end process;

    -- bernoulli output
    o_bernoulli <= '0' when i_activation <= dout else '1';
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        state_counter <= 0;
        dout <= x"030";
    elsif rising_edge(i_clk) then
        if ( i_sync_reset = '1' ) then
            state_counter <= 0;
            dout <= x"030";
        elsif (i_en = '1') then
            if ( state_counter = 61 ) then
                state_counter <= 0;
            else
                state_counter <= state_counter + 1;
            end if;
            dout <= rom(state_counter);
        end if;
    end if;
    end process;

end architecture;